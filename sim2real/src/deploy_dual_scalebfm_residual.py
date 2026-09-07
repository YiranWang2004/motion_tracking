#!/usr/bin/env python3
"""Deploy frozen ScaleBFM plus MAPPO residuals to two isolated G1 bridges."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dual_runtime.deployment_recorder import DualDeploymentRecorder
from dual_runtime.policy_coordinator import DeploymentState
from dual_runtime.runtime_config import shared_control_settings
from dual_runtime.interactive_control import InteractiveDualCoordinator, load_default_command
from dual_runtime.constants import POLICY_JOINT_NAMES
from dual_runtime.scalebfm_standing import DualScaleBFMStanding
from dual_runtime.robot_session import RobotSession, RobotSessionConfig
from dual_runtime.sim_pose_provider import DualSimulationPoseProvider
from dual_runtime.scalebfm_residual_policy import DualScaleBFMResidualPolicy
from dual_runtime.vive_dual_pose import (
    DualViveDeploymentConfig,
    DualVivePoseProvider,
)
from dual_runtime.visualization import (
    DualVisualizationSender,
    publish_runtime_visualization,
)
from omnicontact.runtime import MotionBridgeClient

ACTUATION_CONFIRMATION = "ENABLE_MOTORS"
ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifacts(config_path: Path, raw: dict[str, Any], *, include_residual: bool = True) -> dict[str, Path]:
    artifact_raw = raw["artifacts"]
    directory = resolve(config_path.parent, artifact_raw["directory"])
    manifest_path = resolve(directory, artifact_raw["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "dual_g1_scalebfm_residual_object":
        raise ValueError("artifact manifest task does not match this deploy entry")
    result = {}
    for key in (
        "scalebfm_checkpoint",
        "scalebfm_metadata",
        "scalebfm_mode_table",
        "residual_checkpoint",
        "reference_bundle",
        "kinematics_xml",
    ):
        if key == "residual_checkpoint" and not include_residual:
            continue
        path = resolve(directory, artifact_raw[key])
        entry = manifest.get("files", {}).get(path.name)
        if not path.is_file() or not isinstance(entry, dict):
            raise FileNotFoundError(f"artifact is missing from manifest: {path}")
        actual = sha256(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"artifact checksum mismatch: {path}")
        result[key] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config/g1/dual_scalebfm_residual.yaml")
    )
    parser.add_argument("--scalebfm-only", action="store_true",
                        help="ScaleBFM baseline on sim or Vive/bridges: skip residual actor loading and inference")
    parser.add_argument("--reference-bundle", default=None)
    parser.add_argument("--vive-config", default=None)
    parser.add_argument(
        "--pose-source",
        choices=("vive", "sim"),
        default="vive",
        help="use three Vive trackers or the atomic pose embedded by sim2sim",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--visualization-host", default=None)
    parser.add_argument("--visualization-port", type=int, default=None)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument(
        "--act-robot", choices=("none", "a", "b", "both"), default="none"
    )
    parser.add_argument("--confirm-actuation", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration is not None and args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    if args.act_robot != "none" and args.confirm_actuation != ACTUATION_CONFIRMATION:
        raise SystemExit(
            f"motor output requires --confirm-actuation {ACTUATION_CONFIRMATION}"
        )
    config_path = Path(args.config).expanduser().resolve()
    raw = load_yaml(config_path)
    cpu_affinity = tuple(int(value) for value in raw.get("cpu_affinity", ()))
    if cpu_affinity:
        os.sched_setaffinity(0, set(cpu_affinity))
    control_hz = float(raw.get("control_frequency_hz", 50.0))
    max_inference_time_s = float(raw.get("max_inference_time_s", 0.018))
    max_consecutive_slow_ticks = int(raw.get("max_consecutive_slow_ticks", 3))
    if control_hz <= 0.0:
        raise ValueError("control_frequency_hz must be positive")
    if max_inference_time_s <= 0.0 or max_consecutive_slow_ticks < 1:
        raise ValueError("inference watchdog settings must be positive")
    artifacts = resolve_artifacts(config_path, raw, include_residual=not args.scalebfm_only)
    if args.reference_bundle is not None:
        artifacts["reference_bundle"] = (
            Path(args.reference_bundle).expanduser().resolve()
        )
    device = args.device or str(raw.get("device", "cpu"))
    policy = DualScaleBFMResidualPolicy(
        scalebfm_checkpoint=artifacts["scalebfm_checkpoint"],
        scalebfm_metadata=artifacts["scalebfm_metadata"],
        scalebfm_mode_table=artifacts["scalebfm_mode_table"],
        residual_checkpoint=artifacts.get("residual_checkpoint"),
        residual_enabled=not args.scalebfm_only,
        reference_bundle=artifacts["reference_bundle"],
        kinematics_xml=artifacts["kinematics_xml"],
        device=device,
        inference_precision=str(raw.get("inference_precision", "fp32")),
        control_mode=int(raw.get("control_mode", 7)),
        future_step=int(raw.get("future_step", 5)),
        residual_scale=float(raw.get("residual_scale", 0.10)),
        start_frame=int(raw.get("start_frame", 1)),
        reference_alignment=str(raw.get("reference_alignment", "none")),
        torch_num_threads=int(raw.get("torch_num_threads", 4)),
    )
    initial_reference, _ = policy.reference.frame(policy.start_frame)
    print(
        f"[reference] placement={policy.reference_alignment}, frame={policy.start_frame}; "
        f"box position_w={initial_reference.object_pos_w.tolist()}, "
        f"quaternion_wxyz={initial_reference.object_quat_wxyz.tolist()}; "
        "Vive calibration and measured poses remain in the calibrated world."
    )
    limits_path = resolve(config_path.parent, raw["joint_limits_source"])
    limits = load_yaml(limits_path)
    lower = np.asarray(limits["joint_pos_lowerlimit_lab"], dtype=np.float32)
    upper = np.asarray(limits["joint_pos_upperlimit_lab"], dtype=np.float32)

    if args.pose_source == "sim":
        simulation = raw.get("simulation")
        if not isinstance(simulation, dict):
            raise ValueError("simulation config is required for --pose-source sim")
        sim_robots = simulation.get("robots")
        if not isinstance(sim_robots, dict):
            raise ValueError("simulation.robots must contain robot_a and robot_b")
        provider = DualSimulationPoseProvider()
        session_values = (sim_robots["robot_a"], sim_robots["robot_b"])
        for value in session_values:
            udp = value["udp"]
            for key in ("state_bind_host", "cmd_host"):
                if not ipaddress.ip_address(str(udp[key])).is_loopback:
                    raise ValueError(
                        f"simulation {key} must be loopback, got {udp[key]!r}"
                    )
        vive_path = None
    else:
        vive_path = resolve(config_path.parent, args.vive_config or raw["vive_config"])
        provider = DualVivePoseProvider(
            DualViveDeploymentConfig.load(vive_path, require_object=not args.scalebfm_only),
            require_object=not args.scalebfm_only,
        )
        session_values = (raw["robot_a"], raw["robot_b"])

    control = shared_control_settings(raw)

    def session(
        robot_id: str,
        value: dict[str, Any],
        *,
        observe_sim_pose: bool = False,
    ) -> RobotSession:
        client = MotionBridgeClient(
            value["udp"],
            require_bridge_session=args.pose_source == "vive",
            state_observer=(
                provider.ingest_bridge_state
                if observe_sim_pose and isinstance(provider, DualSimulationPoseProvider)
                else None
            ),
        )
        return RobotSession(
            RobotSessionConfig(
                robot_id=robot_id,
                udp=dict(value["udp"]),
                lower=lower,
                upper=upper,
                kp=policy.kp,
                kd=policy.kd,
                torque_limit=policy.torque_limit,
                max_target_delta=control["phase_target_delta"]["default_pose"],
                damping_kd=control["damping_kd"],
            ),
            client=client,
        )

    default_ticks = round(control["default_pose_duration_s"] * control_hz)
    preflight = raw.get("preflight", {})
    standing_assets = resolve(config_path.parent, control["standing_asset_dir"])
    coordinator = InteractiveDualCoordinator(
        session("a", session_values[0], observe_sim_pose=args.pose_source == "sim"),
        session("b", session_values[1]),
        provider,
        policy,
        standing_policy=DualScaleBFMStanding(policy, load_default_command(standing_assets)),
        default_command=load_default_command(standing_assets),
        input_mode="sim" if args.pose_source == "sim" else "hardware",
        transition_ticks=default_ticks,
        startup_timeout_s=float(
            raw["simulation"].get("startup_timeout_s", 60.0)
            if args.pose_source == "sim" else raw.get("state_startup_timeout_s", 10.0)
        ),
        phase_target_delta=control["phase_target_delta"],
        require_button_release=control["require_button_release"],
        max_tilt_rad=control["max_tilt_rad"],
        task_safety=None if args.scalebfm_only else control["task_safety"],
        enable_a=args.act_robot in {"a", "both"},
        enable_b=args.act_robot in {"b", "both"},
        state_timeout_s=float(raw.get("state_timeout_s", 0.2)),
        pose_timeout_s=float(raw.get("pose_timeout_s", 0.1)),
        max_state_skew_s=float(raw.get("max_state_skew_s", 0.05)),
        default_pose_ticks=default_ticks,
        max_partner_position_error_m=float(
            preflight.get("max_partner_position_error_m", 0.20)
        ),
        max_object_position_error_m=float(
            preflight.get("max_object_position_error_m", 0.20)
        ),
        max_box_size_error_m=float(preflight.get("max_box_size_error_m", 0.03)),
        max_robot_orientation_error_rad=float(
            preflight.get("max_robot_orientation_error_rad", 0.35)
        ),
        max_object_orientation_error_rad=float(
            preflight.get("max_object_orientation_error_rad", 0.35)
        ),
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(
        f"dual ScaleBFM deploy: policy_mode={'scalebfm_only' if args.scalebfm_only else 'scalebfm_residual'} act_robot={args.act_robot} device={device} "
        f"reference={artifacts['reference_bundle']}"
    )
    recorder = None
    if not args.no_record:
        log_dir = resolve(
            config_path.parent,
            args.log_dir
            or raw.get("log_directory", "../../logs/dual_scalebfm_residual"),
        )
        recorder = DualDeploymentRecorder(
            log_dir,
            {
                "config": str(config_path),
                "effective_control": control,
                "configuration": raw,
                "reference": str(artifacts["reference_bundle"]),
                "reference_alignment": str(raw.get("reference_alignment", "none")),
                "start_frame": int(raw.get("start_frame", 1)),
                "pose_source": args.pose_source,
                "standing_policy": "scalebfm_default_pose",
                "object_enabled": not args.scalebfm_only,
                "policy_mode": "scalebfm_only" if args.scalebfm_only else "scalebfm_residual",
                "vive_config": None if vive_path is None else str(vive_path),
                "device": device,
                "act_robot": args.act_robot,
                "control_frequency_hz": control_hz,
            },
        )
    visualization_config = raw.get("visualization", {})
    visualization_sender = None
    if not args.no_visualization and bool(visualization_config.get("enabled", True)):
        visualization_host = args.visualization_host or str(
            visualization_config.get("host", "127.0.0.1")
        )
        visualization_port = int(
            args.visualization_port
            if args.visualization_port is not None
            else visualization_config.get("port", 55204)
        )
        if not 1 <= visualization_port <= 65535:
            raise ValueError("visualization port must be in [1, 65535]")
        visualization_sender = DualVisualizationSender(
            visualization_host, visualization_port
        )
        print(f"visualization stream: udp://{visualization_host}:{visualization_port}")
    ticks = 0
    consecutive_slow_ticks = 0
    try:
        provider.start()
        if args.pose_source == "vive" and not provider.wait_until_ready(10.0):
            raise RuntimeError("Vive provider did not become ready")
        period = 1.0 / control_hz
        start = time.monotonic()
        next_tick = start
        while True:
            if args.duration is not None and time.monotonic() - start >= args.duration:
                if recorder is not None:
                    recorder.metadata["exit_reason"] = "duration_reached"
                break
            result = coordinator.step()
            if recorder is not None:
                recorder.record(coordinator, result)
            publish_runtime_visualization(
                visualization_sender, coordinator, result
            )
            if not result.ok:
                raise RuntimeError(result.reason)
            ticks += 1
            if ticks == 1 and coordinator.last_geometry is not None:
                print(f"preflight geometry: {coordinator.last_geometry}")
            if result.policy_step is not None and (
                result.policy_step.frame == int(raw.get("start_frame", 1))
                or ticks % 50 == 0
            ):
                residual_max = float(np.max(np.abs(result.policy_step.residuals)))
                print(
                    f"ticks={ticks} state={result.state.value} "
                    f"frame={result.policy_step.frame} residual_max={residual_max:.3f} "
                    f"inference_ms={1000.0 * result.policy_step.inference_time_s:.2f}"
                )
            if result.reason not in {"simulation_snapshot_retry", "waiting_for_initial_inputs"}:
                consecutive_slow_ticks = (
                    consecutive_slow_ticks + 1
                    if result.processing_time_s > max_inference_time_s
                    else 0
                )
                if consecutive_slow_ticks >= max_consecutive_slow_ticks:
                    raise RuntimeError(
                        "control processing exceeded the real-time budget for "
                        f"{consecutive_slow_ticks} consecutive ticks"
                    )
            if result.state in {DeploymentState.COMPLETE, DeploymentState.STOPPED}:
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -period:
                next_tick = time.monotonic()
    except BaseException as exc:
        if recorder is not None:
            recorder.metadata["exit_reason"] = f"{type(exc).__name__}: {exc}"
            if coordinator.last_result is not None:
                recorder.record(coordinator, coordinator.last_result)
        raise
    finally:
        try:
            coordinator.close()
        finally:
            provider.stop()
            if visualization_sender is not None:
                visualization_sender.close()
            if recorder is not None:
                saved = recorder.save()
                if saved is not None:
                    print(f"saved deployment log: {saved}")
    print(f"dual ScaleBFM residual deploy complete: ticks={ticks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
