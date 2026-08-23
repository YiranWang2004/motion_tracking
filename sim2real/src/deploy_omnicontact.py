#!/usr/bin/env python3
"""Deploy the carry-box-only OmniContact policy through the existing G1 bridge."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from omnicontact.contracts import PDCommand, TaskGoal
from omnicontact.diagnostics import (
    ObservationHistoryRecorder,
    provider_update_counts,
)
from omnicontact.loco_mode import LocoModePolicy
from omnicontact.perception.pose_udp import UdpPoseReceiverProvider
from omnicontact.perception.vive_pose import ViveDeploymentConfig, VivePoseProvider
from omnicontact.policy import OmniContactCarryPolicy, RobotPolicyState
from omnicontact.runtime import (
    BridgePoseProvider,
    BridgeState,
    CommandLimiter,
    MotionBridgeClient,
    pose_pair_is_valid,
)
from omnicontact.visualization_udp import VisualizationSender
from paths import SIM2REAL_ROOT, controller_config_path, robot_config_path


LOGGER = logging.getLogger("omnicontact.deploy")
ACTUATION_CONFIRMATION = "ENABLE_MOTORS"
DEFAULT_LOG_DIR = SIM2REAL_ROOT.parent / "logs" / "omnicontact"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a mapping")
    return data


def _resolve_config_path(path: str | Path, *, robot: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if len(candidate.parts) > 1:
        return (SIM2REAL_ROOT / candidate).resolve()
    return robot_config_path(robot, candidate).resolve()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing config section {name}")
    return value


def _resolve_prepare_seconds(
    requested: float | None,
    *,
    pose_source: str,
    safety_config: dict[str, Any],
    control_freq: float,
) -> float:
    if requested is not None:
        return float(requested)
    if pose_source == "sim":
        return float(safety_config.get("sim_prepare_seconds", 1.0 / control_freq))
    return float(safety_config["prepare_seconds"])


def _configure_logging(args: argparse.Namespace) -> Path | None:
    """Configure console logging plus one durable, per-run diagnostic log."""
    log_path: Path | None = None
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if not args.no_file_log:
        if args.log_file is not None:
            log_path = Path(args.log_file).expanduser().resolve()
        else:
            log_dir = (
                DEFAULT_LOG_DIR
                if args.log_dir is None
                else Path(args.log_dir).expanduser().resolve()
            )
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            log_path = log_dir / f"deploy_{stamp}_pid{os.getpid()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s "
            "[%(threadName)s]: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    if log_path is None:
        LOGGER.warning("File logging disabled by --no-file-log")
    else:
        LOGGER.info("Session log file: %s", log_path)
    return log_path


def _resolve_history_path(
    args: argparse.Namespace,
    session_log_path: Path | None,
) -> Path | None:
    """Use a same-stem NPZ companion unless structured history was disabled."""
    if args.no_history_file:
        return None
    if args.history_file is not None:
        return Path(args.history_file).expanduser().resolve()
    if session_log_path is None:
        return None
    return session_log_path.with_suffix(".observations.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OmniContact carry-box through motion_tracking's G1 UDP/DDS bridge"
    )
    parser.add_argument("--robot", choices=("g1",), default="g1")
    parser.add_argument("--config", default="omnicontact_carrybox.yaml")
    parser.add_argument("--controller-config", default=None)
    parser.add_argument("--vive-config", default=None)
    parser.add_argument("--pose-source", choices=("sim", "local", "udp"), required=True)
    parser.add_argument(
        "--goal-position",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 0.15),
        metavar=("X", "Y", "Z"),
        help="sim pose mode only: desired box-center position in the MuJoCo world",
    )
    parser.add_argument("--vive-hz", type=float, default=100.0)
    parser.add_argument("--udp-bind", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=15150)
    parser.add_argument(
        "--udp-token",
        default=os.environ.get("OMNICONTACT_POSE_TOKEN")
        or os.environ.get("ROBOJUDO_POSE_TOKEN"),
    )
    parser.add_argument("--allowed-sender-ip", default=None)
    parser.add_argument(
        "--visualization-host",
        default="127.0.0.1",
        help="read-only MuJoCo twin host (independent from the motor UDP ports)",
    )
    parser.add_argument("--visualization-port", type=int, default=55004)
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="disable the read-only reference/ghost visualization stream",
    )
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--prepare-seconds", type=float, default=None)
    parser.add_argument("--max-target-delta", type=float, default=None)
    parser.add_argument("--pose-max-age", type=float, default=None)
    parser.add_argument("--act", action="store_true")
    parser.add_argument("--confirm-actuation", default="")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "directory for an automatically named per-run log "
            f"(default: {DEFAULT_LOG_DIR})"
        ),
    )
    file_log_group = parser.add_mutually_exclusive_group()
    file_log_group.add_argument(
        "--log-file",
        default=None,
        help="write this run to an explicit log file instead of an automatic name",
    )
    file_log_group.add_argument(
        "--no-file-log",
        action="store_true",
        help="disable the default persistent diagnostic log",
    )
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--history-file",
        default=None,
        help=(
            "write exact per-policy-frame observations to this compressed NPZ "
            "(default: same stem as the deploy log)"
        ),
    )
    history_group.add_argument(
        "--no-history-file",
        action="store_true",
        help="disable the default compressed observation-history companion",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.act and args.confirm_actuation != ACTUATION_CONFIRMATION:
        raise SystemExit(
            f"motor output requires --confirm-actuation {ACTUATION_CONFIRMATION}"
        )
    if args.pose_source == "udp" and not args.udp_token:
        raise SystemExit(
            "UDP pose mode requires --udp-token, OMNICONTACT_POSE_TOKEN, "
            "or ROBOJUDO_POSE_TOKEN"
        )
    if args.pose_source != "sim" and not args.vive_config:
        raise SystemExit("--vive-config is required for local and udp pose sources")
    for name in (
        "vive_hz",
        "run_seconds",
        "prepare_seconds",
        "max_target_delta",
        "pose_max_age",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.udp_port <= 65535:
        raise SystemExit("--udp-port must be in [1, 65535]")
    if not 1 <= args.visualization_port <= 65535:
        raise SystemExit("--visualization-port must be in [1, 65535]")


def make_pose_provider(
    args: argparse.Namespace,
    vive_config: ViveDeploymentConfig | None,
):
    if args.pose_source == "sim":
        return BridgePoseProvider()
    assert vive_config is not None
    if args.pose_source == "local":
        return VivePoseProvider(vive_config, poll_hz=args.vive_hz)
    return UdpPoseReceiverProvider(
        args.udp_bind,
        args.udp_port,
        args.udp_token,
        allowed_sender_ip=args.allowed_sender_ip,
    )


def _policy_state(state: BridgeState) -> RobotPolicyState:
    return RobotPolicyState(
        q_lab=state.q_lab,
        dq_lab=state.dq_lab,
        angular_velocity=state.gyro,
    )


def _record_history_safely(
    recorder: ObservationHistoryRecorder | None,
    **values: Any,
) -> None:
    """Keep optional diagnostics from ever becoming a control-loop failure."""
    if recorder is None or recorder.disabled:
        return
    try:
        recorder.record(**values)
    except Exception as exc:
        recorder.disable(exc)
        LOGGER.exception(
            "Structured observation-history recording failed; continuing motor "
            "control without additional structured rows"
        )


def _fresh_pair(provider, *, max_age_s: float, min_confidence: float):
    robot_pose, object_pose = provider.get_poses()
    valid = pose_pair_is_valid(
        robot_pose,
        object_pose,
        max_age_s=max_age_s,
        min_confidence=min_confidence,
    )
    return robot_pose, object_pose, valid


def _wait_for_bridge(client: MotionBridgeClient, timeout_s: float) -> BridgeState:
    LOGGER.info("Waiting for G1 bridge state...")
    deadline = time.monotonic() + timeout_s
    while True:
        state = client.read_next(timeout_s=min(1.0, max(0.0, deadline - time.monotonic())))
        if state is not None:
            LOGGER.info("Connected to G1 bridge at state sequence %d", state.packet_seq)
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"no G1 bridge state received within {timeout_s:.1f}s")


def _zero_torque_until_start(
    client: MotionBridgeClient,
    initial_state: BridgeState,
    *,
    state_timeout_s: float,
) -> BridgeState:
    LOGGER.warning("ZERO TORQUE: press Start to begin original DefaultPose")
    state = initial_state
    while not state.buttons["start"]:
        client.send_zero(state)
        next_state = client.read_next(state_timeout_s)
        if next_state is None:
            raise RuntimeError("lost G1 bridge state while waiting for Start")
        state = next_state
        if client.button_rise.get("stop", False):
            raise KeyboardInterrupt
    LOGGER.warning(
        "PHASE zero_torque->default_pose: Start accepted at state_seq=%d",
        state.packet_seq,
    )
    return state


def _move_to_default(
    client: MotionBridgeClient,
    state: BridgeState,
    policy: OmniContactCarryPolicy,
    *,
    prepare_seconds: float,
    control_freq: float,
    state_timeout_s: float,
) -> BridgeState:
    LOGGER.warning("Moving to original DefaultPose over %.2f seconds", prepare_seconds)
    start_q = state.q_lab.copy()
    steps = max(1, int(round(prepare_seconds * control_freq)))
    for index in range(steps):
        next_state = client.read_next(state_timeout_s)
        if next_state is None:
            raise RuntimeError("lost G1 bridge state during default-pose transition")
        state = next_state
        if client.button_rise.get("stop", False):
            raise KeyboardInterrupt
        alpha = float(index + 1) / float(steps)
        target = start_q * (1.0 - alpha) + policy.default_pose_lab * alpha
        client.send(
            PDCommand(target, policy.default_kp_lab, policy.default_kd_lab),
            enable=1,
            state=state,
        )
    LOGGER.warning(
        "PHASE default_pose->loco_standing: transition complete at state_seq=%d "
        "command_gap_ms=%.3f max_command_gap_ms=%.3f",
        state.packet_seq,
        getattr(client, "last_command_gap_ms", 0.0),
        getattr(client, "max_command_gap_ms", 0.0),
    )
    return state


def _plan_while_holding(
    client: MotionBridgeClient,
    state: BridgeState,
    policy: OmniContactCarryPolicy,
    loco_mode: LocoModePolicy,
    limiter: CommandLimiter,
    robot_pose,
    object_pose,
    goal: TaskGoal,
    *,
    state_timeout_s: float,
) -> BridgeState:
    error: list[BaseException] = []
    plan_start = time.monotonic()

    def plan() -> None:
        try:
            policy.initialize_reference(robot_pose, object_pose, goal)
        except BaseException as exc:
            error.append(exc)
            LOGGER.exception("CFGen initial reference worker failed")

    worker = threading.Thread(target=plan, name="omnicontact-initial-plan", daemon=True)
    worker.start()
    LOGGER.info("Generating initial carry-box reference...")
    while worker.is_alive():
        next_state = client.read_next(state_timeout_s)
        if next_state is None:
            raise RuntimeError("lost G1 bridge state while planning carry reference")
        state = next_state
        if client.button_rise.get("stop", False):
            raise KeyboardInterrupt
        client.send(limiter.apply(loco_mode.compute(state)), enable=1, state=state)
    worker.join()
    if error:
        raise RuntimeError(f"initial carry reference generation failed: {error[0]}") from error[0]
    assert policy.reference is not None
    LOGGER.info(
        "Carry reference ready: frames=%d elapsed_s=%.3f state_seq=%d "
        "command_gap_ms=%.3f max_command_gap_ms=%.3f",
        len(policy.reference["ref_contact"]),
        time.monotonic() - plan_start,
        state.packet_seq,
        getattr(client, "last_command_gap_ms", 0.0),
        getattr(client, "max_command_gap_ms", 0.0),
    )
    return state


def _wait_for_task_start(
    client: MotionBridgeClient,
    state: BridgeState,
    provider,
    policy: OmniContactCarryPolicy,
    loco_mode: LocoModePolicy,
    limiter: CommandLimiter,
    goal: TaskGoal,
    *,
    pose_max_age_s: float,
    min_pose_confidence: float,
    state_timeout_s: float,
) -> BridgeState:
    loco_mode.reset()
    limiter.reset(state.q_lab)
    LOGGER.warning(
        "LocoMode standing active; release the robot, then press A with fresh "
        "robot/object poses to start carry-box"
    )
    last_pose_warning = 0.0
    while True:
        next_state = client.read_next(state_timeout_s)
        if next_state is None:
            raise RuntimeError("lost G1 bridge state while waiting for A")
        state = next_state
        client.send(limiter.apply(loco_mode.compute(state)), enable=1, state=state)
        if client.button_rise.get("stop", False):
            raise KeyboardInterrupt
        if not client.button_rise.get("A", False):
            continue
        robot_pose, object_pose, valid = _fresh_pair(
            provider,
            max_age_s=pose_max_age_s,
            min_confidence=min_pose_confidence,
        )
        if not valid:
            now = time.monotonic()
            if now - last_pose_warning >= 1.0:
                LOGGER.error("A ignored: robot/object pose pair is missing or stale")
                last_pose_warning = now
            continue
        now = time.monotonic()
        valid_count, invalid_count = provider_update_counts(provider)
        LOGGER.warning(
            "A accepted: state_seq=%d robot_pose_age_ms=%.3f "
            "object_pose_age_ms=%.3f robot_confidence=%.3f "
            "object_confidence=%.3f pelvis=%s pelvis_quat_xyzw=%s "
            "object=%s object_quat_xyzw=%s goal=%s "
            "provider_valid=%d provider_invalid=%d",
            state.packet_seq,
            (now - robot_pose.stamp_s) * 1000.0,
            (now - object_pose.stamp_s) * 1000.0,
            robot_pose.confidence,
            object_pose.confidence,
            robot_pose.position_w,
            robot_pose.quaternion_xyzw,
            object_pose.position_w,
            object_pose.quaternion_xyzw,
            goal.position_w,
            valid_count,
            invalid_count,
        )
        return _plan_while_holding(
            client,
            state,
            policy,
            loco_mode,
            limiter,
            robot_pose,
            object_pose,
            goal,
            state_timeout_s=state_timeout_s,
        )


def _run_no_actuation(
    client: MotionBridgeClient,
    state: BridgeState,
    provider,
    policy: OmniContactCarryPolicy,
    goal: TaskGoal,
    *,
    pose_max_age_s: float,
    min_pose_confidence: float,
    state_timeout_s: float,
    run_seconds: float | None,
    history_recorder: ObservationHistoryRecorder | None = None,
) -> BridgeState:
    robot_pose, object_pose, valid = _fresh_pair(
        provider,
        max_age_s=pose_max_age_s,
        min_confidence=min_pose_confidence,
    )
    if not valid:
        raise RuntimeError("pose pair became stale before no-actuation policy initialization")
    policy.initialize_reference(robot_pose, object_pose, goal)
    LOGGER.warning("NO-ACTUATION: evaluating policy without sending any bridge command")
    start = time.monotonic()
    last_log = start
    valid_steps = 0
    stale_steps = 0
    while run_seconds is None or time.monotonic() - start < run_seconds:
        next_state = client.read_next(state_timeout_s)
        if next_state is None:
            raise RuntimeError("lost G1 bridge state in no-actuation mode")
        state = next_state
        robot_pose, object_pose, valid = _fresh_pair(
            provider,
            max_age_s=pose_max_age_s,
            min_confidence=min_pose_confidence,
        )
        if valid:
            result = policy.compute(_policy_state(state), robot_pose, object_pose)
            client.publish_visualization(result.visualization)
            if history_recorder is not None and not history_recorder.disabled:
                _record_history_safely(
                    history_recorder,
                    state=state,
                    provider=provider,
                    policy_frame=policy.frame,
                    event="policy_no_actuation",
                    task_state=result.task_state,
                    pose_pair_valid=True,
                    robot_pose=robot_pose,
                    object_pose=object_pose,
                    observation=result.observation,
                    observation_history=policy.history,
                    policy_action=policy.action_lab,
                    raw_command=result.command,
                )
            policy.advance()
            valid_steps += 1
            if policy.should_replan(object_pose, goal):
                policy.request_replan(robot_pose, object_pose, goal)
        else:
            stale_steps += 1
        now = time.monotonic()
        if now - last_log >= 1.0:
            LOGGER.info(
                "no-act frame=%d valid=%d stale=%d skipped_state=%d",
                policy.frame,
                valid_steps,
                stale_steps,
                client.skipped_packets,
            )
            last_log = now
        if provider.error is not None:
            raise RuntimeError(f"pose provider failed: {provider.error}") from provider.error
    return state


def _run_actuated(
    client: MotionBridgeClient,
    state: BridgeState,
    provider,
    policy: OmniContactCarryPolicy,
    loco_mode: LocoModePolicy,
    goal: TaskGoal,
    limiter: CommandLimiter,
    *,
    pose_max_age_s: float,
    min_pose_confidence: float,
    state_timeout_s: float,
    run_seconds: float | None,
    history_recorder: ObservationHistoryRecorder | None = None,
) -> BridgeState:
    limiter.reset(state.q_lab)
    start = time.monotonic()
    last_warning = 0.0
    last_log = start
    mode = "tracking"
    last_visualization = None
    first_tracking_step = True
    while run_seconds is None or time.monotonic() - start < run_seconds:
        next_state = client.read_next(state_timeout_s)
        if next_state is None:
            robot_pose, object_pose, pose_valid = _fresh_pair(
                provider,
                max_age_s=pose_max_age_s,
                min_confidence=min_pose_confidence,
            )
            if history_recorder is not None and not history_recorder.disabled:
                _record_history_safely(
                    history_recorder,
                    state=state,
                    provider=provider,
                    policy_frame=getattr(policy, "frame", -1),
                    event="bridge_state_timeout",
                    task_state=mode,
                    pose_pair_valid=pose_valid,
                    robot_pose=robot_pose,
                    object_pose=object_pose,
                    command_gap_ms=getattr(client, "last_command_gap_ms", np.nan),
                )
            LOGGER.critical(
                "FAILSAFE damping requested: lost bridge state in tracking; "
                "last_state_seq=%d command_gap_ms=%.3f max_command_gap_ms=%.3f",
                state.packet_seq,
                getattr(client, "last_command_gap_ms", 0.0),
                getattr(client, "max_command_gap_ms", 0.0),
            )
            client.send_damping(state)
            raise RuntimeError("lost G1 bridge state; damping command sent")
        state = next_state
        if client.button_rise.get("stop", False):
            LOGGER.warning(
                "Stop accepted during tracking at state_seq=%d frame=%d",
                state.packet_seq,
                getattr(policy, "frame", -1),
            )
            break
        if mode == "tracking" and provider.error is not None:
            LOGGER.critical(
                "FAILSAFE damping requested: pose provider failed: %r",
                provider.error,
            )
            client.send_damping(state)
            raise RuntimeError(f"pose provider failed: {provider.error}") from provider.error

        if mode == "standing":
            safe_command = limiter.apply(loco_mode.compute(state))
            client.send(
                safe_command,
                enable=1,
                state=state,
                visualization=last_visualization,
            )
            now = time.monotonic()
            if now - last_log >= 1.0:
                LOGGER.info(
                    "state=loco_mode_standing skipped_state=%d target_abs_max=%.3f",
                    client.skipped_packets,
                    float(np.max(np.abs(safe_command.target_pos))),
                )
                last_log = now
            continue

        robot_pose, object_pose, valid = _fresh_pair(
            provider,
            max_age_s=pose_max_age_s,
            min_confidence=min_pose_confidence,
        )
        if not valid:
            hold = limiter.hold(state.q_lab, policy.kp_lab, policy.kd_lab)
            client.send(hold, enable=1, state=state)
            if history_recorder is not None and not history_recorder.disabled:
                _record_history_safely(
                    history_recorder,
                    state=state,
                    provider=provider,
                    policy_frame=policy.frame,
                    event="pose_pair_invalid_hold",
                    task_state="tracking_frozen",
                    pose_pair_valid=False,
                    robot_pose=robot_pose,
                    object_pose=object_pose,
                    safe_command=hold,
                    command_gap_ms=getattr(client, "last_command_gap_ms", np.nan),
                )
            now = time.monotonic()
            if now - last_warning >= 1.0:
                valid_count, invalid_count = provider_update_counts(provider)
                LOGGER.error(
                    "Robot/object pose missing or stale: reference frozen, holding "
                    "measured joints provider_valid=%d provider_invalid=%d",
                    valid_count,
                    invalid_count,
                )
                last_warning = now
            continue

        compute_start = time.monotonic()
        step = policy.compute(_policy_state(state), robot_pose, object_pose)
        compute_ms = (time.monotonic() - compute_start) * 1000.0
        if first_tracking_step:
            LOGGER.warning(
                "PHASE loco_standing->omnicontact_tracking: first policy step "
                "compute_ms=%.3f state_seq=%d command_gap_before_send_ms=%.3f",
                compute_ms,
                state.packet_seq,
                getattr(client, "last_command_gap_ms", 0.0),
            )
            first_tracking_step = False
        elif compute_ms >= 50.0:
            LOGGER.warning(
                "Slow OmniContact policy step: compute_ms=%.3f state_seq=%d frame=%d",
                compute_ms,
                state.packet_seq,
                policy.frame,
            )
        safe_command = limiter.apply(step.command)
        last_visualization = step.visualization
        client.send(
            safe_command,
            enable=1,
            state=state,
            visualization=step.visualization,
        )
        if history_recorder is not None and not history_recorder.disabled:
            _record_history_safely(
                history_recorder,
                state=state,
                provider=provider,
                policy_frame=policy.frame,
                event="policy_tracking",
                task_state=step.task_state,
                pose_pair_valid=True,
                robot_pose=robot_pose,
                object_pose=object_pose,
                observation=step.observation,
                observation_history=policy.history,
                policy_action=policy.action_lab,
                raw_command=step.command,
                safe_command=safe_command,
                policy_compute_ms=compute_ms,
                command_gap_ms=getattr(client, "last_command_gap_ms", np.nan),
            )
        policy.advance()
        if step.task_state == "trajectory_complete":
            mode = "standing"
            loco_mode.reset()
            limiter.reset(safe_command.target_pos)
            LOGGER.warning("CFGen trajectory complete; switched to LocoMode standing")
        elif policy.should_replan(object_pose, goal):
            policy.request_replan(robot_pose, object_pose, goal)

        now = time.monotonic()
        if now - last_log >= 1.0:
            valid_count, invalid_count = provider_update_counts(provider)
            linear_speed = (
                np.nan
                if object_pose.linear_velocity_w is None
                else float(np.linalg.norm(object_pose.linear_velocity_w))
            )
            angular_speed = (
                np.nan
                if object_pose.angular_velocity_w is None
                else float(np.linalg.norm(object_pose.angular_velocity_w))
            )
            LOGGER.info(
                "state=%s frame=%d skipped_state=%d target_abs_max=%.3f "
                "command_gap_ms=%.3f max_command_gap_ms=%.3f "
                "object_linear_speed=%.4f object_angular_speed=%.4f "
                "provider_valid=%d provider_invalid=%d",
                step.task_state,
                policy.frame,
                client.skipped_packets,
                float(np.max(np.abs(safe_command.target_pos))),
                getattr(client, "last_command_gap_ms", 0.0),
                getattr(client, "max_command_gap_ms", 0.0),
                linear_speed,
                angular_speed,
                valid_count,
                invalid_count,
            )
            last_log = now
    return state


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    session_log_path = _configure_logging(args)
    history_path = _resolve_history_path(args, session_log_path)
    LOGGER.info(
        "Session start: pid=%d cwd=%s robot=%s pose_source=%s act=%s "
        "visualization=%s log=%s observation_history=%s",
        os.getpid(),
        Path.cwd(),
        args.robot,
        args.pose_source,
        args.act,
        not args.no_visualization,
        session_log_path,
        history_path,
    )

    omni_path = _resolve_config_path(args.config, robot=args.robot)
    controller_path = (
        controller_config_path(args.robot)
        if args.controller_config is None
        else _resolve_config_path(args.controller_config, robot=args.robot)
    )
    omni_config = _load_yaml(omni_path)
    controller_config = _load_yaml(controller_path)
    pose_config = _section(omni_config, "pose")
    safety_config = _section(omni_config, "safety")
    policy_config = _section(omni_config, "policy")
    control_freq = float(omni_config["control_freq"])
    if control_freq != float(controller_config["control_freq"]):
        raise ValueError(
            f"control frequency mismatch: OmniContact={control_freq}, "
            f"controller={controller_config['control_freq']}"
        )
    LOGGER.info(
        "Resolved configs: omni=%s controller=%s vive=%s control_hz=%.3f",
        omni_path,
        controller_path,
        args.vive_config,
        control_freq,
    )

    asset_dir_value = Path(str(omni_config["asset_dir"]))
    asset_dir = (
        asset_dir_value
        if asset_dir_value.is_absolute()
        else omni_path.parent / asset_dir_value
    )
    pose_max_age_s = float(
        pose_config["max_age_s"] if args.pose_max_age is None else args.pose_max_age
    )
    min_pose_confidence = float(pose_config["min_confidence"])
    wait_timeout_s = float(pose_config["wait_timeout_s"])
    state_timeout_s = float(safety_config["state_timeout_s"])
    prepare_seconds = _resolve_prepare_seconds(
        args.prepare_seconds,
        pose_source=args.pose_source,
        safety_config=safety_config,
        control_freq=control_freq,
    )
    max_target_delta = float(
        safety_config["max_target_delta"]
        if args.max_target_delta is None
        else args.max_target_delta
    )
    LOGGER.info(
        "Safety config: pose_max_age_ms=%.3f state_timeout_s=%.3f "
        "prepare_seconds=%.3f max_target_delta=%.3f damping_kd=%.3f",
        pose_max_age_s * 1000.0,
        state_timeout_s,
        prepare_seconds,
        max_target_delta,
        float(safety_config.get("damping_kd", 8.0)),
    )

    vive_config = (
        None
        if args.pose_source == "sim"
        else ViveDeploymentConfig.load(args.vive_config)
    )
    provider = make_pose_provider(args, vive_config)
    client: MotionBridgeClient | None = None
    visualization_sender = None
    if not args.no_visualization:
        visualization_sender = VisualizationSender(
            args.visualization_host,
            args.visualization_port,
        )
    policy: OmniContactCarryPolicy | None = None
    history_recorder = (
        None
        if history_path is None
        else ObservationHistoryRecorder(
            history_path,
            joint_names=list(controller_config["policy_joint_names"]),
            metadata={
                "pid": os.getpid(),
                "pose_source": args.pose_source,
                "actuation_enabled": bool(args.act),
                "deploy_log": None
                if session_log_path is None
                else session_log_path.as_posix(),
                "omnicontact_config": omni_path.as_posix(),
                "controller_config": controller_path.as_posix(),
                "vive_config": args.vive_config,
                "control_frequency_hz": control_freq,
                "pose_max_age_ms": pose_max_age_s * 1000.0,
                "minimum_pose_confidence": min_pose_confidence,
            },
        )
    )
    if history_recorder is None:
        LOGGER.warning("Structured observation-history logging is disabled")
    else:
        LOGGER.info("Observation history file: %s", history_recorder.path)
    last_state: BridgeState | None = None
    termination_reason = "normal_completion"
    try:
        provider.start()
        client = MotionBridgeClient(
            controller_config["udp"],
            pose_sink=provider if isinstance(provider, BridgePoseProvider) else None,
            visualization_sender=visualization_sender,
        )
        last_state = _wait_for_bridge(client, wait_timeout_s)
        LOGGER.info("Waiting for a complete robot/object pose pair...")
        if not provider.wait_until_ready(wait_timeout_s):
            raise RuntimeError(
                f"no complete robot/object pose pair received within {wait_timeout_s:.1f}s"
            )
        robot_pose, object_pose, valid = _fresh_pair(
            provider,
            max_age_s=pose_max_age_s,
            min_confidence=min_pose_confidence,
        )
        if not valid:
            raise RuntimeError("initial robot/object pose pair is not fresh")
        goal_position = (
            np.asarray(args.goal_position, dtype=np.float32)
            if vive_config is None
            else vive_config.goal_position_w
        )
        LOGGER.warning(
            "Pose sanity: pelvis=%s object=%s goal=%s",
            robot_pose.position_w,
            object_pose.position_w,
            goal_position,
        )

        policy = OmniContactCarryPolicy(
            asset_dir,
            list(controller_config["policy_joint_names"]),
            **policy_config,
        )
        loco_mode = LocoModePolicy(
            asset_dir,
            list(controller_config["policy_joint_names"]),
        )
        limiter = CommandLimiter(
            policy.lower_lab,
            policy.upper_lab,
            max_target_delta,
        )
        goal = TaskGoal(goal_position)
        client.set_carrybox_scene(object_pose, goal)

        if not args.act:
            last_state = _run_no_actuation(
                client,
                last_state,
                provider,
                policy,
                goal,
                pose_max_age_s=pose_max_age_s,
                min_pose_confidence=min_pose_confidence,
                state_timeout_s=state_timeout_s,
                run_seconds=args.run_seconds,
                history_recorder=history_recorder,
            )
            return 0

        LOGGER.critical("MOTOR OUTPUT ENABLED through the G1 bridge")
        last_state = _zero_torque_until_start(
            client,
            last_state,
            state_timeout_s=state_timeout_s,
        )
        last_state = _move_to_default(
            client,
            last_state,
            policy,
            prepare_seconds=prepare_seconds,
            control_freq=control_freq,
            state_timeout_s=state_timeout_s,
        )
        last_state = _wait_for_task_start(
            client,
            last_state,
            provider,
            policy,
            loco_mode,
            limiter,
            goal,
            pose_max_age_s=pose_max_age_s,
            min_pose_confidence=min_pose_confidence,
            state_timeout_s=state_timeout_s,
        )
        last_state = _run_actuated(
            client,
            last_state,
            provider,
            policy,
            loco_mode,
            goal,
            limiter,
            pose_max_age_s=pose_max_age_s,
            min_pose_confidence=min_pose_confidence,
            state_timeout_s=state_timeout_s,
            run_seconds=args.run_seconds,
            history_recorder=history_recorder,
        )
    except KeyboardInterrupt:
        termination_reason = "operator_stop_or_interrupt"
        LOGGER.warning("Interrupted/stopped")
    except Exception:
        termination_reason = "fatal_exception"
        LOGGER.exception("FATAL OmniContact deployment error")
        raise
    finally:
        if termination_reason == "normal_completion":
            if args.act and last_state is not None and last_state.buttons.get("stop", False):
                termination_reason = "stop_button"
            elif args.run_seconds is not None:
                termination_reason = "run_seconds_complete"
        if args.act and client is not None and last_state is not None:
            try:
                final_state = getattr(client, "latest_state", None) or last_state
                final_log = (
                    LOGGER.critical
                    if termination_reason == "fatal_exception"
                    else LOGGER.warning
                )
                final_log(
                    "FINAL damping requested: reason=%s command_count=%d "
                    "last_command_gap_ms=%.3f max_command_gap_ms=%.3f "
                    "last_state_seq=%d",
                    termination_reason,
                    getattr(client, "command_count", 0),
                    getattr(client, "last_command_gap_ms", 0.0),
                    getattr(client, "max_command_gap_ms", 0.0),
                    final_state.packet_seq,
                )
                client.send_damping(
                    final_state,
                    damping_kd=float(
                        _section(omni_config, "safety").get("damping_kd", 8.0)
                    ),
                )
                LOGGER.warning("Final damping command sent")
            except Exception:
                LOGGER.exception("failed to send final damping command")
        if history_recorder is not None:
            try:
                history_recorder.save(termination_reason=termination_reason)
            except Exception:
                LOGGER.exception("failed to save structured observation history")
        if client is not None:
            client.close()
        elif visualization_sender is not None:
            # MotionBridgeClient normally owns the sender after construction.
            # Close it here when startup failed before ownership transferred.
            visualization_sender.close()
        provider.stop()
        if policy is not None:
            policy.close()
        LOGGER.info("Session end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
