#!/usr/bin/env python3
"""Read-only dual-G1 twin with direct Vive preview, bridge state, and replay."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.udp_latest import UDPLatestReceiver  # noqa: E402
from dual_runtime.constants import POLICY_JOINT_NAMES  # noqa: E402
from dual_runtime.vive_dual_pose import DualViveDeploymentConfig  # noqa: E402
from dual_runtime.visualization import DualVisualizationReceiver  # noqa: E402
from dual_runtime.visualization_replay import DualScaleBFMReplay  # noqa: E402
from omnicontact.perception.openvr_tracker import (  # noqa: E402
    OpenVRTrackerReader,
    ViveSample,
)
from omnicontact.perception.vive_pose import (  # noqa: E402
    RigidTransform,
    sample_to_transform,
)
from omnicontact.replay import ReplayClock  # noqa: E402
from omnicontact.viewer_style import configure_camera  # noqa: E402


DEFAULT_BOX_HALF_EXTENTS = np.array([0.15, 0.5, 0.15], dtype=np.float64)


def reference_box_half_extents(path: str | None) -> np.ndarray:
    """Only explicit bundle dimensions override the one-metre display default."""
    if path is None:
        return DEFAULT_BOX_HALF_EXTENTS.copy()
    with np.load(Path(path).expanduser(), allow_pickle=False) as bundle:
        if "training_box_half_extents" in bundle:
            value = np.asarray(bundle["training_box_half_extents"], dtype=np.float64)
        elif "box_size" in bundle:
            value = np.asarray(bundle["box_size"], dtype=np.float64) / 2.0
        else:
            return DEFAULT_BOX_HALF_EXTENTS.copy()
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("bundle box dimensions must be a finite positive 3-vector")
    return value


@dataclass(frozen=True)
class TwinBindings:
    actual_base_qpos: np.ndarray
    actual_joint_qpos: np.ndarray
    reference_base_qpos: np.ndarray
    reference_joint_qpos: np.ndarray
    actual_box_qpos: int
    reference_box_qpos: int
    actual_box_geom: int
    reference_box_geom: int
    tracker_mocap: np.ndarray
    reference_alpha: dict[int, float]


def _object_id(model: mujoco.MjModel, kind: Any, name: str) -> int:
    result = int(mujoco.mj_name2id(model, kind, name))
    if result < 0:
        raise ValueError(f"MuJoCo twin is missing {name}")
    return result


def _qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint])


def _mocap_id(model: mujoco.MjModel, name: str) -> int:
    body = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    result = int(model.body_mocapid[body])
    if result < 0:
        raise ValueError(f"MuJoCo body is not mocap-enabled: {name}")
    return result


def build_bindings(model: mujoco.MjModel) -> TwinBindings:
    def joint_addresses(prefix: str) -> np.ndarray:
        return np.asarray(
            [_qpos_address(model, prefix + name) for name in POLICY_JOINT_NAMES],
            dtype=np.int32,
        )

    return TwinBindings(
        actual_base_qpos=np.asarray(
            [
                _qpos_address(model, "actual_a_floating_base_joint"),
                _qpos_address(model, "actual_b_floating_base_joint"),
            ],
            dtype=np.int32,
        ),
        actual_joint_qpos=np.stack(
            (joint_addresses("actual_a_"), joint_addresses("actual_b_"))
        ),
        reference_base_qpos=np.asarray(
            [
                _qpos_address(model, "reference_a_floating_base_joint"),
                _qpos_address(model, "reference_b_floating_base_joint"),
            ],
            dtype=np.int32,
        ),
        reference_joint_qpos=np.stack(
            (joint_addresses("reference_a_"), joint_addresses("reference_b_"))
        ),
        actual_box_qpos=_qpos_address(model, "actual_box_joint"),
        reference_box_qpos=_qpos_address(model, "reference_box_joint"),
        actual_box_geom=_object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "actual_box_geom"
        ),
        reference_box_geom=_object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "reference_box_geom"
        ),
        tracker_mocap=np.asarray(
            [
                _mocap_id(model, "tracker_a_marker"),
                _mocap_id(model, "tracker_b_marker"),
                _mocap_id(model, "tracker_object_marker"),
            ],
            dtype=np.int32,
        ),
        reference_alpha=_robot_copy_alpha(model, "reference_"),
    )


def _robot_copy_alpha(model: mujoco.MjModel, prefix: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        if body_name is not None and body_name.startswith(prefix):
            result[geom_id] = float(model.geom_rgba[geom_id, 3])
    return result


def _style_robot_copy(
    model: mujoco.MjModel,
    prefix: str,
    color: tuple[float, float, float] | None,
    *,
    alpha: float,
    group: int,
) -> None:
    color_array = None if color is None else np.asarray(color, dtype=np.float32)
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        if body_name is None or not body_name.startswith(prefix):
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is not None and "_frame_" in geom_name:
            model.geom_rgba[geom_id, 3] = alpha
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            model.geom_group[geom_id] = group
            continue
        if color_array is not None:
            brightness = float(
                np.clip(np.mean(model.geom_rgba[geom_id, :3]), 0.2, 1.0)
            )
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id, :3] = np.clip(
                0.35 * color_array + 0.65 * brightness, 0.0, 1.0
            )
            model.geom_rgba[geom_id, 3] = alpha
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        model.geom_group[geom_id] = group


def configure_twin_appearance(model: mujoco.MjModel) -> None:
    # Match the single-robot viewer: measured robots keep their native G1
    # materials, while reference/target copies use warm translucent colors.
    _style_robot_copy(model, "actual_a_", None, alpha=1.0, group=0)
    _style_robot_copy(model, "actual_b_", None, alpha=1.0, group=0)
    _style_robot_copy(model, "reference_a_", (1.0, 0.90, 0.05), alpha=0.24, group=1)
    _style_robot_copy(model, "reference_b_", (1.0, 0.32, 0.02), alpha=0.24, group=1)


def _set_reference_visibility(
    model: mujoco.MjModel,
    alpha_by_geom: dict[int, float],
    visible: bool,
) -> None:
    for geom_id, alpha in alpha_by_geom.items():
        model.geom_rgba[geom_id, 3] = alpha if visible else 0.0


def _set_actual_robot_visibility(model: mujoco.MjModel, visible: bool) -> None:
    for prefix in ("actual_a_", "actual_b_"):
        for geom_id, alpha in _robot_copy_alpha(model, prefix).items():
            model.geom_rgba[geom_id, 3] = alpha if visible else 0.0


def load_twin(xml_path: str | Path) -> tuple[mujoco.MjModel, mujoco.MjData, TwinBindings]:
    path = Path(xml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    configure_twin_appearance(model)
    data = mujoco.MjData(model)
    bindings = build_bindings(model)
    return model, data, bindings


def decode_bridge_joint_state(data: Any) -> np.ndarray | None:
    if not isinstance(data, dict) or "q" not in data:
        return None
    try:
        q = np.asarray(data["q"], dtype=np.float32).reshape(29)
    except (TypeError, ValueError):
        return None
    return q.copy() if np.all(np.isfinite(q)) else None


def apply_bridge_joint_state(
    data: mujoco.MjData,
    bindings: TwinBindings,
    robot_index: int,
    q: np.ndarray,
) -> None:
    data.qpos[bindings.actual_joint_qpos[int(robot_index)]] = np.asarray(
        q, dtype=np.float64
    ).reshape(29)


def _set_freejoint(data: mujoco.MjData, address: int, pose_wxyz: Any) -> None:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    data.qpos[address : address + 7] = pose


def _set_mocap(data: mujoco.MjData, mocap_id: int, pose_wxyz: Any) -> None:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    data.mocap_pos[mocap_id] = pose[:3]
    data.mocap_quat[mocap_id] = pose[3:]


def _transform_wxyz(transform: RigidTransform) -> np.ndarray:
    quaternion = transform.quaternion_xyzw
    return np.concatenate(
        (transform.position, quaternion[[3, 0, 1, 2]])
    )


def _load_default_pose(path: str | Path) -> np.ndarray:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        values = np.asarray(raw["default_angles_lab"], dtype=np.float64).reshape(
            len(POLICY_JOINT_NAMES)
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid OmniContact default pose {config_path}: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"default pose contains non-finite values: {config_path}")
    return values


def initialize_default_pose(
    data: mujoco.MjData,
    bindings: TwinBindings,
    default_pose: Any,
) -> None:
    values = np.asarray(default_pose, dtype=np.float64).reshape(
        len(POLICY_JOINT_NAMES)
    )
    data.qpos[bindings.actual_joint_qpos] = values
    data.qpos[bindings.reference_joint_qpos] = values


def apply_vive_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: TwinBindings,
    config: DualViveDeploymentConfig,
    samples: dict[str, ViveSample | None],
    last_tracker_transforms: list[RigidTransform | None],
) -> tuple[bool, bool, bool]:
    """Apply raw Tracker markers and calibrated actual poses independently."""

    serials = (
        config.robot_a_tracker_serial,
        config.robot_b_tracker_serial,
        config.object_tracker_serial,
    )
    extrinsics = (
        config.robot_a_tracker_to_pelvis,
        config.robot_b_tracker_to_pelvis,
        config.object_tracker_to_object,
    )
    fresh = tuple(samples.get(serial) is not None for serial in serials)
    for index, (serial, extrinsic) in enumerate(zip(serials, extrinsics)):
        sample = samples.get(serial)
        if sample is not None:
            last_tracker_transforms[index] = config.world_from_steamvr.compose(
                sample_to_transform(sample)
            )
        world_from_tracker = last_tracker_transforms[index]
        if world_from_tracker is None:
            continue
        _set_mocap(
            data,
            int(bindings.tracker_mocap[index]),
            _transform_wxyz(world_from_tracker),
        )
        calibrated_pose = _transform_wxyz(
            world_from_tracker.compose(extrinsic)
        )
        if index < 2:
            _set_freejoint(
                data, int(bindings.actual_base_qpos[index]), calibrated_pose
            )
        else:
            _set_freejoint(data, bindings.actual_box_qpos, calibrated_pose)
    return fresh


def apply_visualization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: TwinBindings,
    packet: dict[str, Any],
    *,
    ghost_mode: str = "reference",
    apply_actual: bool = True,
    apply_tracker_markers: bool = True,
    box_half_extents: np.ndarray | None = None,
) -> None:
    if ghost_mode not in {"reference", "target"}:
        raise ValueError("ghost_mode must be 'reference' or 'target'")
    _set_reference_visibility(model, bindings.reference_alpha, True)
    actual = packet["actual"]
    reference = packet["reference"]
    policy = packet["policy"]
    actual_bases = np.asarray(actual["robot_base_wxyz"], dtype=np.float64)
    reference_bases = np.asarray(reference["robot_base_wxyz"], dtype=np.float64)
    for index in range(2):
        if apply_actual:
            _set_freejoint(
                data, int(bindings.actual_base_qpos[index]), actual_bases[index]
            )
        ghost_base = (
            reference_bases[index]
            if ghost_mode == "reference"
            else data.qpos[
                int(bindings.actual_base_qpos[index]) :
                int(bindings.actual_base_qpos[index]) + 7
            ]
        )
        _set_freejoint(data, int(bindings.reference_base_qpos[index]), ghost_base)
        if apply_tracker_markers:
            _set_mocap(
                data, int(bindings.tracker_mocap[index]), actual_bases[index]
            )
    ghost_joints = (
        reference["joint_pos"]
        if ghost_mode == "reference"
        else policy["target_joint_pos"]
    )
    data.qpos[bindings.reference_joint_qpos] = np.asarray(
        ghost_joints, dtype=np.float64
    )
    if apply_actual:
        _set_freejoint(data, bindings.actual_box_qpos, actual["object_wxyz"])
    _set_freejoint(data, bindings.reference_box_qpos, reference["object_wxyz"])
    if apply_tracker_markers:
        _set_mocap(
            data, int(bindings.tracker_mocap[2]), actual["object_wxyz"]
        )
    # The reference packet carries the active CFGen bundle dimensions. Vive
    # config/actual packet dimensions must not replace the display default.
    extents = (reference.get("box_half_extents", DEFAULT_BOX_HALF_EXTENTS)
               if box_half_extents is None else box_half_extents)
    model.geom_size[bindings.actual_box_geom, :3] = extents
    model.geom_size[bindings.reference_box_geom, :3] = extents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml-path",
        default=str(ROOT / "config/g1/assets/dual_scalebfm_twin.xml"),
    )
    parser.add_argument("--state-host-a", default="0.0.0.0")
    parser.add_argument("--state-port-a", type=int, default=55003)
    parser.add_argument("--state-host-b", default="0.0.0.0")
    parser.add_argument("--state-port-b", type=int, default=55103)
    parser.add_argument("--visualization-host", default="127.0.0.1")
    parser.add_argument("--visualization-port", type=int, default=55204)
    parser.add_argument("--stream-timeout", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--vive-config",
        default=str(ROOT / "config/g1/omnicontact_vive_dual.json"),
        help="three-Tracker calibration used by the direct live preview",
    )
    parser.add_argument(
        "--no-vive",
        action="store_true",
        help="disable direct OpenVR input and use policy visualization poses",
    )
    parser.add_argument(
        "--default-pose-config",
        default=str(ROOT / "config/g1/omnicontact/OmniContact.yaml"),
        help="OmniContact YAML containing default_angles_lab",
    )
    parser.add_argument("--replay-log", default=None)
    parser.add_argument("--reference-bundle", default=None)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-start-frame", type=int, default=0)
    parser.add_argument("--replay-loop", action="store_true")
    parser.add_argument("--replay-paused", action="store_true")
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="hide both measured G1 meshes while keeping ghosts, boxes and frames",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="load and validate the MJCF without opening a window or UDP sockets",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0.0 or args.stream_timeout <= 0.0:
        parser.error("--fps and --stream-timeout must be positive")
    if args.replay_speed <= 0.0 or args.replay_start_frame < 0:
        parser.error("replay speed must be positive and start frame non-negative")
    for name in ("state_port_a", "state_port_b", "visualization_port"):
        value = int(getattr(args, name))
        if not 0 <= value <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 65535]")
    if args.replay_log is None and args.visualization_port == 0:
        parser.error("live mode requires a non-zero visualization port")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model, data, bindings = load_twin(args.xml_path)
    initial_extents = reference_box_half_extents(args.reference_bundle)
    explicit_extents = initial_extents if args.reference_bundle is not None else None
    model.geom_size[bindings.actual_box_geom, :3] = initial_extents
    model.geom_size[bindings.reference_box_geom, :3] = initial_extents
    initialize_default_pose(
        data, bindings, _load_default_pose(args.default_pose_config)
    )
    if args.no_robot:
        _set_actual_robot_visibility(model, False)
    _set_reference_visibility(model, bindings.reference_alpha, False)
    mujoco.mj_forward(model, data)
    if args.check_model:
        print(
            f"dual ScaleBFM twin model OK: nq={model.nq} nv={model.nv} "
            f"bodies={model.nbody} geoms={model.ngeom} mocap={model.nmocap}"
        )
        return 0

    replay = None
    clock = None
    visualization_receiver = None
    state_receivers: list[UDPLatestReceiver | None] = [None, None]
    vive_config = None
    vive_reader = None
    if args.replay_log is not None:
        replay = DualScaleBFMReplay(
            args.replay_log, reference_bundle=args.reference_bundle
        )
        if args.replay_start_frame >= replay.frame_count:
            raise ValueError(
                f"--replay-start-frame must be below {replay.frame_count}"
            )
        clock = ReplayClock(
            replay,
            speed=args.replay_speed,
            start_frame=args.replay_start_frame,
            paused=args.replay_paused,
            loop=args.replay_loop,
        )
        print(
            f"dual ScaleBFM replay: {replay.path} frames={replay.frame_count} "
            f"duration={replay.duration_s:.3f}s speed={args.replay_speed:g}x"
        )
    else:
        if not args.no_vive:
            vive_config = DualViveDeploymentConfig.load(args.vive_config)
            vive_reader = OpenVRTrackerReader(
                (
                    vive_config.robot_a_tracker_serial,
                    vive_config.robot_b_tracker_serial,
                    vive_config.object_tracker_serial,
                )
            )
        visualization_receiver = DualVisualizationReceiver(
            args.visualization_host, args.visualization_port
        )
        visualization_receiver.start()
        endpoints = (
            (args.state_host_a, args.state_port_a),
            (args.state_host_b, args.state_port_b),
        )
        for index, (host, port) in enumerate(endpoints):
            if port:
                receiver = UDPLatestReceiver(host, port)
                receiver.start()
                state_receivers[index] = receiver
        print(
            "live dual twin: "
            f"A={args.state_host_a}:{args.state_port_a} "
            f"B={args.state_host_b}:{args.state_port_b} "
            f"visualization={args.visualization_host}:{args.visualization_port}"
        )
    print(
        "Native G1 materials: measured A/B. Transparent yellow/orange: A/B ghost. "
        "Brown/orange boxes: measured/reference."
    )
    if replay is not None or vive_reader is None:
        print(
            "Blue/purple/orange pyramids show reconstructed A/B/object poses; "
            "red/green/blue axes are X/Y/Z."
        )
    else:
        print(
            "Blue/purple/orange pyramids are raw A/B/object Trackers; "
            "G1 and box poses include the configured mounting extrinsics."
        )
    print(
        "Keys: G=reference/final-PD-target ghost, Space/P=pause, N/B=step, "
        "A/D=slower/faster, R=restart."
    )

    ghost_mode = ["reference"]
    last_state_seq: list[int | None] = [None, None]
    last_state_recv_ns: list[int | None] = [None, None]
    last_visual_seq: int | None = None
    last_visual_recv_ns: int | None = None
    last_replay_index: int | None = None
    last_status = time.monotonic()
    appearance_refresh = [False]
    reference_visible = [False]
    last_tracker_transforms: list[RigidTransform | None] = [None, None, None]
    last_tracker_seen: list[float | None] = [None, None, None]
    last_tracker_refresh = time.monotonic()

    def on_key(keycode: int) -> None:
        if keycode in (ord("G"), ord("g")):
            ghost_mode[0] = "target" if ghost_mode[0] == "reference" else "reference"
            appearance_refresh[0] = True
            print(f"ghost mode: {ghost_mode[0]}")
        if clock is None:
            return
        if keycode in (ord(" "), ord("P"), ord("p")):
            clock.toggle_pause()
        elif keycode in (ord("N"), ord("n")):
            clock.step(1)
        elif keycode in (ord("B"), ord("b")):
            clock.step(-1)
        elif keycode in (ord("A"), ord("a")):
            clock.shift_speed(-1)
        elif keycode in (ord("D"), ord("d")):
            clock.shift_speed(1)
        elif keycode in (ord("R"), ord("r")):
            clock.restart(paused=False)

    frame_period = 1.0 / args.fps
    try:
        if vive_reader is not None:
            devices = vive_reader.start()
            assert vive_config is not None
            print(
                "direct Vive preview: "
                f"A={vive_config.robot_a_tracker_serial} "
                f"B={vive_config.robot_b_tracker_serial} "
                f"object={vive_config.object_tracker_serial}"
            )
            print(f"Detected trackers: {', '.join(sorted(devices))}")
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=on_key,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            configure_camera(viewer.cam, lookat=(0.8, 0.45, 0.8), distance=3.2)
            while viewer.is_running():
                loop_start = time.monotonic()
                with viewer.lock():
                    if replay is not None:
                        assert clock is not None
                        replay_index = clock.update()
                        if replay_index != last_replay_index or appearance_refresh[0]:
                            packet = replay.visualization_packet(replay_index)
                            apply_visualization(
                                model,
                                data,
                                bindings,
                                packet,
                                ghost_mode=ghost_mode[0],
                                box_half_extents=explicit_extents,
                            )
                            for robot_index in range(2):
                                apply_bridge_joint_state(
                                    data,
                                    bindings,
                                    robot_index,
                                    replay.arrays["q"][replay_index, robot_index],
                                )
                            data.time = float(replay.elapsed_s[replay_index])
                            last_replay_index = replay_index
                            appearance_refresh[0] = False
                            reference_visible[0] = True
                    else:
                        assert visualization_receiver is not None
                        if vive_reader is not None:
                            assert vive_config is not None
                            serials = (
                                vive_config.robot_a_tracker_serial,
                                vive_config.robot_b_tracker_serial,
                                vive_config.object_tracker_serial,
                            )
                            samples = vive_reader.read_all(serials)
                            fresh = apply_vive_samples(
                                model,
                                data,
                                bindings,
                                vive_config,
                                samples,
                                last_tracker_transforms,
                            )
                            tracker_now = time.monotonic()
                            for index, is_fresh in enumerate(fresh):
                                if is_fresh:
                                    last_tracker_seen[index] = tracker_now
                            if not all(fresh) and tracker_now - last_tracker_refresh >= 1.0:
                                vive_reader.refresh_devices()
                                last_tracker_refresh = tracker_now
                        visual = visualization_receiver.read_latest(with_meta=True)
                        if visual is not None and (
                            visual.seq != last_visual_seq or appearance_refresh[0]
                        ):
                            apply_visualization(
                                model,
                                data,
                                bindings,
                                visual.data,
                                ghost_mode=ghost_mode[0],
                                box_half_extents=explicit_extents,
                                apply_actual=vive_reader is None,
                                apply_tracker_markers=vive_reader is None,
                            )
                            last_visual_seq = visual.seq
                            last_visual_recv_ns = visual.recv_time_ns
                            appearance_refresh[0] = False
                            reference_visible[0] = True
                        for robot_index, receiver in enumerate(state_receivers):
                            if receiver is None:
                                continue
                            state_packet = receiver.read_latest_data(with_meta=True)
                            if state_packet is None or state_packet.seq == last_state_seq[robot_index]:
                                continue
                            q = decode_bridge_joint_state(state_packet.data)
                            if q is not None:
                                apply_bridge_joint_state(
                                    data, bindings, robot_index, q
                                )
                                last_state_seq[robot_index] = state_packet.seq
                                last_state_recv_ns[robot_index] = state_packet.recv_time_ns
                        if (
                            reference_visible[0]
                            and last_visual_recv_ns is not None
                            and (time.perf_counter_ns() - last_visual_recv_ns) * 1.0e-9
                            > args.stream_timeout
                        ):
                            _set_reference_visibility(
                                model, bindings.reference_alpha, False
                            )
                            reference_visible[0] = False
                    mujoco.mj_forward(model, data)
                viewer.sync()
                now = time.monotonic()
                if now - last_status >= 1.0:
                    if replay is not None:
                        assert clock is not None
                        state = "paused" if clock.paused else "playing"
                        print(
                            f"replay {clock.index + 1}/{replay.frame_count} "
                            f"t={replay.elapsed_s[clock.index]:.2f}s {state} "
                            f"{clock.speed:g}x ghost={ghost_mode[0]}"
                        )
                    else:
                        now_ns = time.perf_counter_ns()
                        ages = [
                            None if stamp is None else (now_ns - stamp) * 1.0e-9
                            for stamp in (*last_state_recv_ns, last_visual_recv_ns)
                        ]
                        labels = ("A", "B", "visual")
                        status = " ".join(
                            f"{label}={'missing' if age is None else f'{age * 1000:.0f}ms'}"
                            + (
                                "!"
                                if age is None or age > args.stream_timeout
                                else ""
                            )
                            for label, age in zip(labels, ages)
                        )
                        if vive_reader is not None:
                            tracker_ages = [
                                None if stamp is None else now - stamp
                                for stamp in last_tracker_seen
                            ]
                            status += " " + " ".join(
                                f"{label}="
                                + (
                                    "missing!"
                                    if age is None
                                    else f"{age * 1000:.0f}ms"
                                    + ("!" if age > args.stream_timeout else "")
                                )
                                for label, age in zip(
                                    ("viveA", "viveB", "viveObj"), tracker_ages
                                )
                            )
                        print(f"streams {status} ghost={ghost_mode[0]}")
                    last_status = now
                delay = frame_period - (time.monotonic() - loop_start)
                if delay > 0.0:
                    time.sleep(delay)
    finally:
        if vive_reader is not None:
            vive_reader.stop()
        if visualization_receiver is not None:
            visualization_receiver.close()
        for receiver in state_receivers:
            if receiver is not None:
                receiver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
