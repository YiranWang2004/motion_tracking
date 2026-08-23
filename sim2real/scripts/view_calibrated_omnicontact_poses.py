#!/usr/bin/env python3
"""Read-only MuJoCo view of calibrated Vive poses in the OmniContact scene.

The script loads the same carry-box XML used by ``sim2sim.py`` and overlays
the world, pelvis, box, and both Tracker coordinate frames.  Vive data is
transformed with the deployment calibration, while no command packet is sent.
Robot joints use OmniContact's default pose unless ``--state-port`` is given.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

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


def _axis_geoms(prefix: str) -> str:
    return f'''\
      <geom name="{prefix}_axis_x" type="capsule" fromto="0 0 0 {AXIS_LENGTH} 0 0" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
      <geom name="{prefix}_axis_y" type="capsule" fromto="0 0 0 0 {AXIS_LENGTH} 0" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
      <geom name="{prefix}_axis_z" type="capsule" fromto="0 0 0 0 0 {AXIS_LENGTH}" size="{AXIS_RADIUS}" contype="0" conaffinity="0" rgba="0.10 0.38 1 1"/>'''


def _expanded_xml(xml_path: Path) -> str:
    """Inject marker assets/bodies without modifying the repository XML."""
    source = xml_path.read_text(encoding="utf-8")
    assets = f'''\
        <mesh name="calib_tracker_pyramid" vertex="{_pyramid_vertices()}" face="0 2 1 0 1 3 1 2 3 2 0 3"/>
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


def _load_joint_setup(model: mujoco.MjModel, config_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    controller = yaml.safe_load((config_dir / "bridge_omnicontact.yaml").read_text())
    names = list(controller["policy_joint_names"])
    defaults = yaml.safe_load((config_dir / "omnicontact" / "OmniContact.yaml").read_text())["default_angles_lab"]
    if len(names) != len(defaults):
        raise RuntimeError("policy_joint_names and default_angles_lab have different lengths")
    addresses = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo model is missing joint {name!r}")
        addresses.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addresses, dtype=int), np.asarray(defaults, dtype=float)


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


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vive-config", default="config/g1/omnicontact_vive.json")
    parser.add_argument("--allow-unconfirmed", action="store_true", help="仅用于可视化，允许 calibration_confirmed=false")
    parser.add_argument("--xml-path", default="config/g1/assets/omnicontact_carry_box.xml")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=None, help="optional read-only G1 state UDP port")
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    parser.add_argument("--no-robot", action="store_true", help="hide the G1 mesh")
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.stale_timeout <= 0:
        parser.error("--fps and --stale-timeout must be positive")
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
    xml_path = (root / args.xml_path).resolve()
    reader = OpenVRTrackerReader((config.robot_tracker_serial, config.object_tracker_serial))
    receiver = None
    if args.state_port is not None:
        receiver = UDPLatestReceiver(args.state_host, args.state_port)
        receiver.start()

    temp = tempfile.NamedTemporaryFile("w", suffix=".xml", prefix="calibrated_view_", dir=xml_path.parent, delete=False, encoding="utf-8")
    temp_path = Path(temp.name)
    try:
        temp.write(_expanded_xml(xml_path))
        temp.close()
        model = mujoco.MjModel.from_xml_path(str(temp_path))
        data = mujoco.MjData(model)
        robot_root = _qpos_addr(model, "floating_base_joint")
        box_root = _qpos_addr(model, "box")
        tracker_robot = _qpos_addr(model, "calib_robot_tracker_free")
        tracker_object = _qpos_addr(model, "calib_object_tracker_free")
        # Joint order/defaults belong to config/g1, while xml_path normally
        # points one level deeper at config/g1/assets.
        joint_addr, default_angles = _load_joint_setup(model, root / "config" / "g1")
        data.qpos[joint_addr] = default_angles
        if args.no_robot:
            _set_robot_visibility(model, False)
        mujoco.mj_forward(model, data)
        devices = reader.start()
        print(f"Calibrated OmniContact viewer: robot={config.robot_tracker_serial}, object={config.object_tracker_serial}")
        print(f"Detected trackers: {', '.join(sorted(devices))}")
        print("Blue/orange pyramids are trackers; red/green/blue axes are X/Y/Z. Read-only viewer.")
        frame_period = 1.0 / args.fps
        last_state_seq = None
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = (0.8, 0.0, 0.8)
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                started = time.monotonic()
                samples = reader.read_all((config.robot_tracker_serial, config.object_tracker_serial))
                rs, os = samples.get(config.robot_tracker_serial), samples.get(config.object_tracker_serial)
                if rs is not None:
                    wrt = config.world_from_steamvr.compose(sample_to_transform(rs))
                    pelvis = wrt.compose(config.robot_tracker_to_pelvis)
                    _set_transform_qpos(data, tracker_robot, wrt)
                    _set_mocap(model, data, "calib_pelvis_frame", pelvis)
                    if not args.no_robot:
                        _set_transform_qpos(data, robot_root, pelvis)
                if os is not None:
                    wot = config.world_from_steamvr.compose(sample_to_transform(os))
                    box = wot.compose(config.object_tracker_to_object)
                    _set_transform_qpos(data, tracker_object, wot)
                    _set_mocap(model, data, "calib_box_frame", box)
                    _set_transform_qpos(data, box_root, box)
                if receiver is not None:
                    packet = receiver.read_latest_data(with_meta=True)
                    if packet is not None and packet.seq != last_state_seq:
                        q = np.asarray(packet.data.get("q", []), dtype=float).reshape(-1)
                        if q.size == joint_addr.size and np.all(np.isfinite(q)):
                            data.qpos[joint_addr] = q
                            last_state_seq = packet.seq
                mujoco.mj_forward(model, data)
                viewer.sync()
                delay = frame_period - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
    finally:
        reader.stop()
        if receiver is not None:
            receiver.close()
        try:
            temp_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
