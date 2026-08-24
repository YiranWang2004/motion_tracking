#!/usr/bin/env python3
"""Read-only MuJoCo view of calibrated Vive poses in the OmniContact scene.

The script loads the same carry-box XML used by ``sim2sim.py`` and overlays
the world, pelvis, box, and both Tracker coordinate frames.  Vive data is
transformed with the deployment calibration, while no command packet is sent.
Robot joints use the bridge's read-only state mirror by default and retain the
OmniContact default pose until the first state packet arrives.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
import threading
import time
import warnings
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.udp_latest import UDPLatestReceiver
from omnicontact.perception.openvr_tracker import OpenVRTrackerReader
from omnicontact.perception.vive_pose import (
    RigidTransform,
    ViveDeploymentConfig,
    sample_to_transform,
)
from omnicontact.replay import (
    OmniContactReplayLog,
    ReplayClock,
    format_replay_progress,
)
from omnicontact.visualization_udp import VisualizationReceiver


AXIS_LENGTH = 0.22
AXIS_RADIUS = 0.006
PYRAMID_SIDE = 0.10
PYRAMID_HEIGHT = 0.10


def _pyramid_vertices() -> str:
    h = PYRAMID_SIDE * 0.5
    y0 = PYRAMID_SIDE * np.sqrt(3.0) / 6.0
    y1 = PYRAMID_SIDE * np.sqrt(3.0) / 3.0
    vertices = ((-h, -y0, 0.0), (h, -y0, 0.0), (0.0, y1, 0.0), (0.0, 0.0, -PYRAMID_HEIGHT))
    return " ".join(f"{v:.9g}" for point in vertices for v in point)


def _pyramid_faces() -> str:
    # Counter-clockwise when viewed from outside. MuJoCo culls backfaces, so
    # inward-facing winding makes parts of this closed marker disappear.
    return "0 1 2  0 3 1  1 3 2  2 3 0"


def _axis_geoms(prefix: str) -> str:
    return f'''\
      <geom name="{prefix}_axis_x" type="capsule" fromto="0 0 0 {AXIS_LENGTH} 0 0" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
      <geom name="{prefix}_axis_y" type="capsule" fromto="0 0 0 0 {AXIS_LENGTH} 0" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
      <geom name="{prefix}_axis_z" type="capsule" fromto="0 0 0 0 0 {AXIS_LENGTH}" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.10 0.38 1 1"/>'''


def _expanded_xml(xml_path: Path) -> str:
    """Inject marker assets/bodies without modifying the repository XML."""
    source = xml_path.read_text(encoding="utf-8")
    assets = f'''\
        <mesh name="calib_tracker_pyramid" vertex="{_pyramid_vertices()}" face="{_pyramid_faces()}"/>
        <material name="calib_tracker_robot" rgba="0.10 0.45 1 1"/>
        <material name="calib_tracker_object" rgba="1 0.35 0.06 1"/>'''
    source = source.replace("    </asset>", assets + "\n    </asset>", 1)
    bodies = f'''\
        <geom name="calib_world_axis_x" type="capsule" fromto="0 0 0 0.5 0 0" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
        <geom name="calib_world_axis_y" type="capsule" fromto="0 0 0 0 0.5 0" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
        <geom name="calib_world_axis_z" type="capsule" fromto="0 0 0 0 0 0.5" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.10 0.38 1 1"/>
        <geom name="calib_world_origin" type="sphere" pos="0 0 0" size="0.018" contype="0" conaffinity="0" rgba="1 0.85 0.1 1"/>
        <body name="calib_robot_tracker">
            <freejoint name="calib_robot_tracker_free"/>
            <geom name="calib_robot_tracker_geom" type="mesh" mesh="calib_tracker_pyramid" material="calib_tracker_robot" contype="0" conaffinity="0"/>
{_axis_geoms("calib_robot_tracker")}
        </body>
        <body name="calib_object_tracker">
            <freejoint name="calib_object_tracker_free"/>
            <geom name="calib_object_tracker_geom" type="mesh" mesh="calib_tracker_pyramid" material="calib_tracker_object" contype="0" conaffinity="0"/>
{_axis_geoms("calib_object_tracker")}
        </body>
        <body name="calib_pelvis_frame" mocap="true">
{_axis_geoms("calib_pelvis")}
        </body>
        <body name="calib_box_frame" mocap="true">
{_axis_geoms("calib_box")}
        </body>
        <light name="calib_fill_light" pos="-2 1 3" dir="2 -1 -3" diffuse="0.8 0.85 1" specular="0.15 0.15 0.15"/>'''
    source = source.replace("    </worldbody>", bodies + "\n    </worldbody>", 1)
    return source


def _qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"missing free joint {joint_name!r}")
    return int(model.jnt_qposadr[jid])


def _set_transform_qpos(data: mujoco.MjData, address: int, transform: RigidTransform) -> None:
    data.qpos[address : address + 3] = transform.position
    q = transform.quaternion_xyzw
    data.qpos[address + 3 : address + 7] = (q[3], q[0], q[1], q[2])


def _inverse_transform(transform: RigidTransform) -> RigidTransform:
    rotation = Rotation.from_quat(transform.quaternion_xyzw).inv()
    return RigidTransform(
        rotation.apply(-transform.position),
        rotation.as_quat(),
    )


def _logged_transform(
    replay: OmniContactReplayLog,
    index: int,
    position_field: str,
    quaternion_field: str,
) -> RigidTransform | None:
    if position_field not in replay.arrays or quaternion_field not in replay.arrays:
        return None
    position = np.asarray(replay.arrays[position_field][index], dtype=np.float64)
    quaternion = np.asarray(replay.arrays[quaternion_field][index], dtype=np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        return None
    if float(np.linalg.norm(quaternion)) < 1e-9:
        return None
    return RigidTransform(position, quaternion)


def _replay_vive_config(
    replay: OmniContactReplayLog,
    fallback: ViveDeploymentConfig,
) -> tuple[ViveDeploymentConfig, bool]:
    raw = replay.metadata.get("vive_calibration")
    if not isinstance(raw, dict):
        return fallback, False
    try:
        return (
            ViveDeploymentConfig(
                robot_tracker_serial=raw["robot_tracker_serial"],
                object_tracker_serial=raw["object_tracker_serial"],
                world_from_steamvr=RigidTransform.from_dict(
                    raw["world_from_steamvr"], "world_from_steamvr"
                ),
                robot_tracker_to_pelvis=RigidTransform.from_dict(
                    raw["robot_tracker_to_pelvis"], "robot_tracker_to_pelvis"
                ),
                object_tracker_to_object=RigidTransform.from_dict(
                    raw["object_tracker_to_object"], "object_tracker_to_object"
                ),
                object_half_extents_m=raw["object_half_extents_m"],
                goal_position_w=raw["goal_position_w"],
                calibration_confirmed=True,
            ),
            True,
        )
    except (KeyError, TypeError, ValueError):
        return fallback, False


def _set_mocap(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, transform: RigidTransform) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"missing body {body_name!r}")
    mocap_id = int(model.body_mocapid[bid])
    if mocap_id < 0:
        raise RuntimeError(f"body {body_name!r} is not a mocap body")
    data.mocap_pos[mocap_id] = transform.position
    q = transform.quaternion_xyzw
    data.mocap_quat[mocap_id] = (q[3], q[0], q[1], q[2])


def _transform_to_xyz_rpy(transform: RigidTransform) -> np.ndarray:
    """Return XYZ meters and fixed-axis XYZ roll/pitch/yaw in degrees."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rpy_deg = Rotation.from_quat(transform.quaternion_xyzw).as_euler(
            "xyz", degrees=True
        )
    return np.concatenate((transform.position, rpy_deg))


def _xyz_rpy_to_transform(values: np.ndarray) -> RigidTransform:
    values = np.asarray(values, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(values)):
        raise ValueError("tracker-to-pelvis slider values must be finite")
    quaternion_xyzw = Rotation.from_euler(
        "xyz", values[3:], degrees=True
    ).as_quat()
    return RigidTransform(values[:3], quaternion_xyzw)


def _save_robot_tracker_to_pelvis(
    config_path: Path,
    transform: RigidTransform,
) -> None:
    """Atomically update only robot_tracker_to_pelvis in the JSON config."""
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Vive config {config_path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(
        raw.get("robot_tracker_to_pelvis"), dict
    ):
        raise ValueError("Vive config is missing robot_tracker_to_pelvis")
    raw["robot_tracker_to_pelvis"] = {
        "position_m": [float(value) for value in transform.position],
        "quaternion_xyzw": [
            float(value) for value in transform.quaternion_xyzw
        ],
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(raw, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, config_path.stat().st_mode & 0o777)
        os.replace(temporary_path, config_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


class RobotTrackerToPelvisTuner:
    """Thread-owned Tk sliders with a lock-protected transform snapshot."""

    def __init__(
        self,
        initial_transform: RigidTransform,
        *,
        translation_range_m: float,
        save_requested: threading.Event,
    ) -> None:
        self._values = _transform_to_xyz_rpy(initial_transform)
        self._translation_range_m = max(
            float(translation_range_m),
            float(np.max(np.abs(self._values[:3]))) + 0.05,
        )
        self._save_requested = save_requested
        self._lock = threading.Lock()
        self._status = "未保存：拖动滑条实时预览；按 S 写入配置"
        self._status_revision = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="robot-tracker-to-pelvis-tuner",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("timed out while opening tracker calibration sliders")
        if self._error is not None:
            raise RuntimeError(
                f"cannot open tracker calibration sliders: {self._error}"
            ) from self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def transform(self) -> RigidTransform:
        with self._lock:
            values = self._values.copy()
        return _xyz_rpy_to_transform(values)

    def mark_saved(self, message: str) -> None:
        with self._lock:
            self._status = message
            self._status_revision += 1

    def _set_value(self, index: int, raw_value: str) -> None:
        with self._lock:
            self._values[index] = float(raw_value)
            self._status = "有未保存修改：按 S 写入配置"
            self._status_revision += 1

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title("robot_tracker_to_pelvis 标定")
            root.resizable(False, False)
            tk.Label(
                root,
                text=(
                    "绝对变换 ^Tracker T_pelvis\n"
                    "XYZ: 米；RPY: 度（固定轴 X-Y-Z）"
                ),
                justify="left",
            ).pack(anchor="w", padx=12, pady=(10, 4))

            labels = (
                "X (m)",
                "Y (m)",
                "Z (m)",
                "Roll (deg)",
                "Pitch (deg)",
                "Yaw (deg)",
            )
            for index, label in enumerate(labels):
                is_translation = index < 3
                limit = self._translation_range_m if is_translation else 180.0
                resolution = 0.001 if is_translation else 0.1
                slider = tk.Scale(
                    root,
                    label=label,
                    from_=-limit,
                    to=limit,
                    resolution=resolution,
                    orient=tk.HORIZONTAL,
                    length=480,
                )
                slider.set(float(self._values[index]))
                slider.configure(
                    command=lambda value, i=index: self._set_value(i, value)
                )
                slider.pack(fill="x", padx=12)

            status_variable = tk.StringVar(value=self._status)
            tk.Label(
                root,
                textvariable=status_variable,
                fg="#8b2500",
                justify="left",
            ).pack(anchor="w", padx=12, pady=(6, 10))
            root.bind_all("<KeyPress-s>", lambda _event: self._save_requested.set())
            root.bind_all("<KeyPress-S>", lambda _event: self._save_requested.set())

            last_status_revision = -1

            def poll_state() -> None:
                nonlocal last_status_revision
                if self._stop.is_set():
                    root.destroy()
                    return
                with self._lock:
                    status = self._status
                    revision = self._status_revision
                if revision != last_status_revision:
                    status_variable.set(status)
                    last_status_revision = revision
                root.after(50, poll_state)

            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.after(50, poll_state)
            self._ready.set()
            root.mainloop()
        except BaseException as exc:
            self._error = exc
            self._ready.set()


def _load_joint_setup(
    model: mujoco.MjModel, config_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controller = yaml.safe_load((config_dir / "bridge_omnicontact.yaml").read_text())
    policy_names = list(controller["policy_joint_names"])
    mujoco_names = list(controller.get("mujoco_joint_names", policy_names))
    defaults = yaml.safe_load((config_dir / "omnicontact" / "OmniContact.yaml").read_text())["default_angles_lab"]
    if len(policy_names) != len(defaults):
        raise RuntimeError("policy_joint_names and default_angles_lab have different lengths")
    addresses = []
    for name in policy_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo model is missing joint {name!r}")
        addresses.append(int(model.jnt_qposadr[jid]))
    ghost_addresses = []
    for name in mujoco_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ghost_{name}")
        if jid < 0:
            raise RuntimeError(f"MuJoCo model is missing ghost joint {name!r}")
        ghost_addresses.append(int(model.jnt_qposadr[jid]))
    return (
        np.asarray(addresses, dtype=int),
        np.asarray(defaults, dtype=float),
        np.asarray(ghost_addresses, dtype=int),
    )


def _dof_addr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"missing joint {joint_name!r}")
    return int(model.jnt_dofadr[jid])


def _set_freejoint_pose(data: mujoco.MjData, address: int, pose: np.ndarray) -> None:
    data.qpos[address : address + 7] = pose


def _reference_visual_alpha(model: mujoco.MjModel) -> dict[int, float]:
    alpha_by_geom: dict[int, float] = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and (body_name.startswith("ref_") or body_name.startswith("ghost_")):
            alpha_by_geom[geom_id] = float(model.geom_rgba[geom_id, 3])
    return alpha_by_geom


def _set_reference_visibility(
    model: mujoco.MjModel,
    alpha_by_geom: dict[int, float],
    visible: bool,
) -> None:
    for geom_id, alpha in alpha_by_geom.items():
        model.geom_rgba[geom_id, 3] = alpha if visible else 0.0


def _apply_visualization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    visualization: dict,
    visual_mocap_ids: dict[str, int],
    ghost_object_qpos: int,
    ghost_robot_qpos: int,
    ghost_joint_qpos: np.ndarray,
    reference_alpha: dict[int, float],
    contact_geom_ids: np.ndarray,
) -> None:
    scene = visualization.get("scene")
    if isinstance(scene, dict):
        for name in ("start_plane_wxyz", "goal_plane_wxyz"):
            mocap_id = visual_mocap_ids[name]
            data.mocap_pos[mocap_id] = scene[name][:3]
            data.mocap_quat[mocap_id] = scene[name][3:7]
    reference = visualization.get("reference")
    if not isinstance(reference, dict):
        return
    _set_reference_visibility(model, reference_alpha, True)
    for name in (
        "left_wrist_wxyz", "right_wrist_wxyz", "torso_wxyz",
        "left_ankle_wxyz", "right_ankle_wxyz",
    ):
        mocap_id = visual_mocap_ids[name]
        data.mocap_pos[mocap_id] = reference[name][:3]
        data.mocap_quat[mocap_id] = reference[name][3:7]
    _set_freejoint_pose(data, ghost_object_qpos, reference["object_wxyz"])
    if "ghost_base_wxyz" in reference:
        _set_freejoint_pose(data, ghost_robot_qpos, reference["ghost_base_wxyz"])
    if "ghost_dof_pos" in reference:
        values = np.asarray(reference["ghost_dof_pos"], dtype=np.float32).reshape(-1)
        if values.size == ghost_joint_qpos.size:
            data.qpos[ghost_joint_qpos] = values
    contact_on = np.asarray(reference["contact"], dtype=np.float32).reshape(4) >= 0.5
    for geom_id, active in zip(contact_geom_ids, contact_on):
        if geom_id >= 0:
            model.geom_rgba[geom_id] = (
                [1.0, 0.0, 0.0, 0.7] if active else [1.0, 1.0, 0.0, 0.7]
            )


def _set_robot_visibility(model: mujoco.MjModel, visible: bool) -> None:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        return
    for geom_id, body_id in enumerate(model.geom_bodyid):
        current = int(body_id)
        while current >= 0:
            if current == pelvis_id:
                model.geom_rgba[geom_id, 3] = 1.0 if visible else 0.0
                break
            parent = int(model.body_parentid[current])
            if parent == current:
                break
            current = parent


def _apply_replay_frame(
    replay: OmniContactReplayLog,
    index: int,
    *,
    config: ViveDeploymentConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_root: int,
    box_root: int,
    box_dof: int,
    tracker_robot: int,
    tracker_object: int,
    joint_qpos: np.ndarray,
    joint_dof: np.ndarray,
    joint_columns: np.ndarray,
    show_robot: bool,
) -> tuple[bool, bool]:
    """Apply measured runtime state; return exactness for robot/object Trackers."""
    arrays = replay.arrays
    q = np.asarray(arrays["q_lab"][index], dtype=np.float64)[joint_columns]
    dq = np.asarray(arrays["dq_lab"][index], dtype=np.float64)[joint_columns]
    if np.all(np.isfinite(q)):
        data.qpos[joint_qpos] = q
    if np.all(np.isfinite(dq)):
        data.qvel[joint_dof] = dq

    pelvis = _logged_transform(
        replay, index, "robot_position_w", "robot_quaternion_xyzw"
    )
    box = _logged_transform(
        replay, index, "object_position_w", "object_quaternion_xyzw"
    )
    if pelvis is not None:
        _set_mocap(model, data, "calib_pelvis_frame", pelvis)
        if show_robot:
            _set_transform_qpos(data, robot_root, pelvis)
    if box is not None:
        _set_transform_qpos(data, box_root, box)
        _set_mocap(model, data, "calib_box_frame", box)
        linear_velocity = np.asarray(
            arrays["object_linear_velocity_w"][index], dtype=np.float64
        )
        angular_velocity = np.asarray(
            arrays["object_angular_velocity_w"][index], dtype=np.float64
        )
        if np.all(np.isfinite(linear_velocity)):
            data.qvel[box_dof : box_dof + 3] = linear_velocity
        if np.all(np.isfinite(angular_velocity)):
            data.qvel[box_dof + 3 : box_dof + 6] = angular_velocity

    robot_tracker = _logged_transform(
        replay,
        index,
        "robot_tracker_position_w",
        "robot_tracker_quaternion_xyzw",
    )
    object_tracker = _logged_transform(
        replay,
        index,
        "object_tracker_position_w",
        "object_tracker_quaternion_xyzw",
    )
    robot_tracker_exact = robot_tracker is not None
    object_tracker_exact = object_tracker is not None
    if robot_tracker is None and pelvis is not None:
        robot_tracker = pelvis.compose(
            _inverse_transform(config.robot_tracker_to_pelvis)
        )
    if object_tracker is None and box is not None:
        object_tracker = box.compose(
            _inverse_transform(config.object_tracker_to_object)
        )
    if robot_tracker is not None:
        _set_transform_qpos(data, tracker_robot, robot_tracker)
    if object_tracker is not None:
        _set_transform_qpos(data, tracker_object, object_tracker)
    data.time = float(replay.elapsed_s[index])
    mujoco.mj_forward(model, data)
    return robot_tracker_exact, object_tracker_exact


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vive-config", default="config/g1/omnicontact_vive.json")
    parser.add_argument("--allow-unconfirmed", action="store_true", help="仅用于可视化，允许 calibration_confirmed=false")
    parser.add_argument("--xml-path", default="config/g1/assets/omnicontact_carry_box.xml")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--replay-log",
        default=None,
        help="回放 deploy_*.log 或 deploy_*.observations.npz；不会连接 OpenVR/UDP",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-start-frame", type=int, default=0)
    parser.add_argument("--replay-loop", action="store_true")
    parser.add_argument("--replay-paused", action="store_true")
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument(
        "--state-port",
        type=int,
        default=55003,
        help="read-only G1 state mirror UDP port (bridge default: 55003); use 0 to disable",
    )
    parser.add_argument("--visualization-host", default="127.0.0.1")
    parser.add_argument("--visualization-port", type=int, default=55004)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    parser.add_argument("--no-robot", action="store_true", help="hide the G1 mesh")
    parser.add_argument(
        "--tune-robot-tracker-to-pelvis",
        action="store_true",
        help="open live XYZ/RPY sliders; press S to atomically save the transform",
    )
    parser.add_argument(
        "--tune-translation-range",
        type=float,
        default=0.5,
        help="absolute XYZ slider range in meters (default: 0.5)",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.stale_timeout <= 0:
        parser.error("--fps and --stale-timeout must be positive")
    if args.tune_translation_range <= 0:
        parser.error("--tune-translation-range must be positive")
    if args.replay_speed <= 0:
        parser.error("--replay-speed must be positive")
    if args.replay_start_frame < 0:
        parser.error("--replay-start-frame must be non-negative")
    if args.replay_log is not None and args.tune_robot_tracker_to_pelvis:
        parser.error("Tracker calibration sliders are unavailable during log replay")
    for name in ("state_port", "visualization_port"):
        port = int(getattr(args, name))
        if not 0 <= port <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 65535]")
    if not args.no_visualization and args.visualization_port == 0:
        parser.error("--visualization-port must be non-zero unless --no-visualization is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.vive_config).resolve()
    try:
        config = ViveDeploymentConfig.load(config_path)
    except ValueError as exc:
        if not args.allow_unconfirmed or "calibration_confirmed must be true" not in str(exc):
            raise
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        raw["calibration_confirmed"] = True
        config = ViveDeploymentConfig(
            robot_tracker_serial=raw["robot_tracker_serial"], object_tracker_serial=raw["object_tracker_serial"],
            world_from_steamvr=RigidTransform.from_dict(raw["world_from_steamvr"], "world_from_steamvr"),
            robot_tracker_to_pelvis=RigidTransform.from_dict(raw["robot_tracker_to_pelvis"], "robot_tracker_to_pelvis"),
            object_tracker_to_object=RigidTransform.from_dict(raw["object_tracker_to_object"], "object_tracker_to_object"),
            object_half_extents_m=raw["object_half_extents_m"], goal_position_w=raw["goal_position_w"], calibration_confirmed=True,
        )
        print("Warning: calibration_confirmed=false; displaying the current values in read-only mode.")
    replay = (
        None
        if args.replay_log is None
        else OmniContactReplayLog.load(args.replay_log)
    )
    replay_has_embedded_calibration = False
    if replay is not None:
        config, replay_has_embedded_calibration = _replay_vive_config(
            replay, config
        )
    if replay is not None and args.replay_start_frame >= replay.frame_count:
        raise ValueError(
            f"--replay-start-frame must be below {replay.frame_count}"
        )
    xml_path = (root / args.xml_path).resolve()
    reader = (
        None
        if replay is not None
        else OpenVRTrackerReader(
            (config.robot_tracker_serial, config.object_tracker_serial)
        )
    )
    receiver = None
    if replay is None and args.state_port:
        receiver = UDPLatestReceiver(args.state_host, args.state_port)
        receiver.start()
    visualization_receiver = None
    tuner = None
    save_requested = threading.Event()
    if replay is None and not args.no_visualization:
        visualization_receiver = VisualizationReceiver(
            args.visualization_host, args.visualization_port
        )
        visualization_receiver.start()

    temp = tempfile.NamedTemporaryFile("w", suffix=".xml", prefix="calibrated_view_", dir=xml_path.parent, delete=False, encoding="utf-8")
    temp_path = Path(temp.name)
    try:
        temp.write(_expanded_xml(xml_path))
        temp.close()
        model = mujoco.MjModel.from_xml_path(str(temp_path))
        data = mujoco.MjData(model)
        robot_root = _qpos_addr(model, "floating_base_joint")
        box_root = _qpos_addr(model, "box")
        box_dof = _dof_addr(model, "box")
        tracker_robot = _qpos_addr(model, "calib_robot_tracker_free")
        tracker_object = _qpos_addr(model, "calib_object_tracker_free")
        # Joint order/defaults belong to config/g1, while xml_path normally
        # points one level deeper at config/g1/assets.
        joint_addr, default_angles, ghost_joint_addr = _load_joint_setup(
            model, root / "config" / "g1"
        )
        bridge_config = yaml.safe_load(
            (root / "config" / "g1" / "bridge_omnicontact.yaml").read_text(
                encoding="utf-8"
            )
        )
        viewer_joint_names = list(bridge_config["policy_joint_names"])
        joint_dof_addr = np.asarray(
            [
                _dof_addr(model, name)
                for name in viewer_joint_names
            ],
            dtype=int,
        )
        replay_joint_columns = (
            None
            if replay is None
            else replay.joint_columns(viewer_joint_names)
        )
        ghost_object_qpos = _qpos_addr(model, "ghost_box_joint")
        ghost_robot_qpos = _qpos_addr(model, "ghost_floating_base_joint")
        visual_mocap_ids = {
            name: int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])
            for name in (
                "plane_1_holder", "plane_2_holder", "ref_l_wrist_frame",
                "ref_r_wrist_frame", "ref_torso_frame", "ref_l_ankle_frame",
                "ref_r_ankle_frame",
            )
        }
        visual_mocap_ids = {
            "start_plane_wxyz": visual_mocap_ids["plane_1_holder"],
            "goal_plane_wxyz": visual_mocap_ids["plane_2_holder"],
            "left_wrist_wxyz": visual_mocap_ids["ref_l_wrist_frame"],
            "right_wrist_wxyz": visual_mocap_ids["ref_r_wrist_frame"],
            "torso_wxyz": visual_mocap_ids["ref_torso_frame"],
            "left_ankle_wxyz": visual_mocap_ids["ref_l_ankle_frame"],
            "right_ankle_wxyz": visual_mocap_ids["ref_r_ankle_frame"],
        }
        contact_geom_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in (
                    "ref_l_ankle_mesh", "ref_r_ankle_mesh",
                    "ref_l_rubber_hand", "ref_r_rubber_hand",
                )
            ],
            dtype=int,
        )
        reference_alpha = _reference_visual_alpha(model)
        _set_reference_visibility(model, reference_alpha, False)
        data.qpos[joint_addr] = default_angles
        if args.no_robot:
            _set_robot_visibility(model, False)
        mujoco.mj_forward(model, data)
        if replay is None:
            assert reader is not None
            devices = reader.start()
            print(f"Calibrated OmniContact viewer: robot={config.robot_tracker_serial}, object={config.object_tracker_serial}")
            print(f"Detected trackers: {', '.join(sorted(devices))}")
            print("Blue/orange pyramids are trackers; red/green/blue axes are X/Y/Z. Read-only viewer.")
        else:
            print(f"OmniContact runtime replay: {replay.path}")
            print(
                f"schema=v{replay.schema_version}, frames={replay.frame_count}, "
                f"duration={replay.duration_s:.3f}s, speed={args.replay_speed:g}x"
            )
            if replay.has_recorded_trackers:
                print("Tracker source: exact world-frame transforms recorded during deployment.")
            else:
                print(
                    "WARNING: old log has no Tracker transforms; pyramids are reconstructed "
                    "from pelvis/box plus "
                    + (
                        "the calibration embedded in the log."
                        if replay_has_embedded_calibration
                        else "the selected current Vive config."
                    ),
                    file=sys.stderr,
                )
            print("Replay keys: Space/P=pause, N=next frame, B=previous frame, R=restart.")
        if args.tune_robot_tracker_to_pelvis:
            tuner = RobotTrackerToPelvisTuner(
                config.robot_tracker_to_pelvis,
                translation_range_m=args.tune_translation_range,
                save_requested=save_requested,
            )
            tuner.start()
            print(
                "Calibration sliders opened. Press S in either window to save "
                f"robot_tracker_to_pelvis to {config_path}"
            )

        replay_clock = (
            None
            if replay is None
            else ReplayClock(
                replay,
                speed=args.replay_speed,
                start_frame=args.replay_start_frame,
                paused=args.replay_paused,
                loop=args.replay_loop,
            )
        )
        progress_refresh = [True]

        def on_key(keycode: int) -> None:
            if tuner is not None and keycode in (ord("S"), ord("s")):
                save_requested.set()
            if replay_clock is None:
                return
            if keycode in (ord(" "), ord("P"), ord("p")):
                replay_clock.toggle_pause()
                progress_refresh[0] = True
            elif keycode in (ord("N"), ord("n")):
                replay_clock.step(1)
                progress_refresh[0] = True
            elif keycode in (ord("B"), ord("b")):
                replay_clock.step(-1)
                progress_refresh[0] = True
            elif keycode in (ord("R"), ord("r")):
                replay_clock.restart(paused=False)
                progress_refresh[0] = True

        frame_period = 1.0 / args.fps
        last_state_seq = None
        last_visual_seq = None
        last_reference_time = 0.0
        last_world_from_robot_tracker = None
        last_replay_index = None
        last_progress_width = 0
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=on_key,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.lookat[:] = (0.8, 0.0, 0.8)
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                started = time.monotonic()
                if replay is not None:
                    assert replay_clock is not None
                    assert replay_joint_columns is not None
                    replay_index = replay_clock.update(started)
                    if replay_index != last_replay_index or progress_refresh[0]:
                        robot_exact, object_exact = _apply_replay_frame(
                            replay,
                            replay_index,
                            config=config,
                            model=model,
                            data=data,
                            robot_root=robot_root,
                            box_root=box_root,
                            box_dof=box_dof,
                            tracker_robot=tracker_robot,
                            tracker_object=tracker_object,
                            joint_qpos=joint_addr,
                            joint_dof=joint_dof_addr,
                            joint_columns=replay_joint_columns,
                            show_robot=not args.no_robot,
                        )
                        tracker_source = (
                            "recorded"
                            if robot_exact and object_exact
                            else "reconstructed/held"
                        )
                        progress = format_replay_progress(
                            replay,
                            replay_index,
                            paused=replay_clock.paused,
                            speed=replay_clock.speed,
                        ) + f" trackers={tracker_source}"
                        padding = " " * max(0, last_progress_width - len(progress))
                        print("\r" + progress + padding, end="", flush=True)
                        last_progress_width = len(progress)
                        last_replay_index = replay_index
                        progress_refresh[0] = False
                    viewer.sync()
                    delay = frame_period - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
                    continue

                assert reader is not None
                samples = reader.read_all((config.robot_tracker_serial, config.object_tracker_serial))
                rs, os = samples.get(config.robot_tracker_serial), samples.get(config.object_tracker_serial)
                if rs is not None:
                    last_world_from_robot_tracker = config.world_from_steamvr.compose(
                        sample_to_transform(rs)
                    )
                if last_world_from_robot_tracker is not None:
                    tracker_to_pelvis = (
                        config.robot_tracker_to_pelvis
                        if tuner is None
                        else tuner.transform()
                    )
                    pelvis = last_world_from_robot_tracker.compose(
                        tracker_to_pelvis
                    )
                    _set_transform_qpos(
                        data, tracker_robot, last_world_from_robot_tracker
                    )
                    _set_mocap(model, data, "calib_pelvis_frame", pelvis)
                    if not args.no_robot:
                        _set_transform_qpos(data, robot_root, pelvis)
                if os is not None:
                    wot = config.world_from_steamvr.compose(sample_to_transform(os))
                    box = wot.compose(config.object_tracker_to_object)
                    _set_transform_qpos(data, tracker_object, wot)
                    _set_mocap(model, data, "calib_box_frame", box)
                    _set_transform_qpos(data, box_root, box)
                if tuner is not None and save_requested.is_set():
                    save_requested.clear()
                    transform_to_save = tuner.transform()
                    try:
                        _save_robot_tracker_to_pelvis(
                            config_path, transform_to_save
                        )
                    except (OSError, ValueError) as exc:
                        message = f"保存失败：{exc}"
                        tuner.mark_saved(message)
                        print(message, file=sys.stderr)
                    else:
                        values = _transform_to_xyz_rpy(transform_to_save)
                        message = (
                            "已保存 robot_tracker_to_pelvis: "
                            f"xyz={values[:3].round(4).tolist()} m, "
                            f"rpy={values[3:].round(2).tolist()} deg"
                        )
                        tuner.mark_saved(message)
                        print(f"{message} -> {config_path}")
                if receiver is not None:
                    packet = receiver.read_latest_data(with_meta=True)
                    if packet is not None and packet.seq != last_state_seq:
                        q = np.asarray(packet.data.get("q", []), dtype=float).reshape(-1)
                        if q.size == joint_addr.size and np.all(np.isfinite(q)):
                            data.qpos[joint_addr] = q
                            last_state_seq = packet.seq
                if visualization_receiver is not None:
                    packet = visualization_receiver.read_latest(with_meta=True)
                    if packet is not None and packet.seq != last_visual_seq:
                        visualization = packet.data
                        _apply_visualization(
                            model,
                            data,
                            visualization,
                            visual_mocap_ids,
                            ghost_object_qpos,
                            ghost_robot_qpos,
                            ghost_joint_addr,
                            reference_alpha,
                            contact_geom_ids,
                        )
                        last_visual_seq = packet.seq
                        if isinstance(visualization.get("reference"), dict):
                            last_reference_time = started
                if (
                    last_reference_time > 0.0
                    and started - last_reference_time > args.stale_timeout
                ):
                    _set_reference_visibility(model, reference_alpha, False)
                mujoco.mj_forward(model, data)
                viewer.sync()
                delay = frame_period - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
    finally:
        if replay is not None:
            print()
        if tuner is not None:
            tuner.close()
        if reader is not None:
            reader.stop()
        if receiver is not None:
            receiver.close()
        if visualization_receiver is not None:
            visualization_receiver.close()
        try:
            temp_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
