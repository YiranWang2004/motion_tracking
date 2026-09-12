#!/usr/bin/env python3
"""G1 onboard single-side ScaleBFM/residual deployment with Vive-only pose input."""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
from pathlib import Path
import select
import sys
import time

# Load PyTorch's OpenMP runtime before MuJoCo/other native libraries. On older
# aarch64 glibc, loading it later can exhaust the static TLS block at startup.
import torch  # noqa: F401
import numpy as np

from dual_runtime.onboard_config import (
    load_yaml, load_onboard_config, channel_endpoints, resolve, sha256, load_artifacts, fingerprint,
)
from dual_runtime.onboard_network import LatestChannel, decode_pose
from dual_runtime.onboard_policy import OnboardPolicy, OnboardStanding
from dual_runtime.onboard_team import TeamState
from dual_runtime.interactive_control import load_default_command
from dual_runtime.runtime_config import shared_control_settings
from dual_runtime.robot_session import RobotSession, RobotSessionConfig
from omnicontact.contracts import PDCommand
from omnicontact.runtime import MotionBridgeClient


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/g1/onboard_scalebfm.yaml")
    parser.add_argument("--robot", choices=("a", "b"), required=True)
    parser.add_argument("--reference-bundle")
    parser.add_argument("--artifact-directory")
    parser.add_argument("--device")
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--confirm-actuation", default="")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--log-dir", default="logs/onboard_scalebfm")
    return parser


def main():
    args = build_parser().parse_args()
    if args.actuate and args.confirm_actuation != "ENABLE_MOTORS":
        raise SystemExit("motor output requires --confirm-actuation ENABLE_MOTORS")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("duration must be positive")
    path = Path(args.config).expanduser().resolve()
    config = load_onboard_config(path)
    base_path = resolve(path.parent, config["policy_config"])
    raw = load_yaml(base_path)
    control = shared_control_settings(raw)
    net = config["network"]
    safety = config["runtime"]
    for key, value in safety.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid runtime setting {key}")
    if float(raw["control_frequency_hz"]) != 50.:
        raise ValueError("onboard reference requires 50 Hz")
    if raw.get("reference_alignment") != "motion_world":
        raise ValueError("onboard requires a shared motion_world reference")
    if safety["max_command_gap_s"] >= .2 or safety["max_processing_s"] >= safety["max_command_gap_s"]:
        raise ValueError("control deadlines must precede the bridge's 200 ms watchdog")
    if not 0 < net["max_clock_rtt_s"] <= .02:
        raise ValueError("max_clock_rtt_s must be in (0, .02]")
    files, contract = load_artifacts(resolve(path.parent, args.artifact_directory or config["artifact_directory"]))
    if args.reference_bundle:
        files["reference_bundle"] = Path(args.reference_bundle).expanduser().resolve()
    if not np.isclose(float(contract["residual_scale"]), float(raw["residual_scale"])):
        raise ValueError("residual scale differs from checkpoint contract")
    calibration = resolve(path.parent, config["vive_config"])
    calibration_id = sha256(calibration)
    limits_path = resolve(base_path.parent, raw["joint_limits_source"])
    standing_dir = resolve(base_path.parent, control["standing_asset_dir"])
    # Include every control-affecting shared setting, but not local IP/device.
    identity = fingerprint({**files, "calibration": calibration, "limits": limits_path,
                            "default": standing_dir / "DefaultPose.yaml",
                            "joint_mapping": standing_dir / "OmniContact.yaml"},
                           dict(policy=raw, control=control, safety=safety, contract=contract,
                                clock_rtt=net["max_clock_rtt_s"], actuation=args.actuate,
                                transport=net["transport"],
                                acceleration_backend=config.get("acceleration", {}).get("backend", "pytorch")))
    affinity = config.get("cpu_affinity", [])
    if affinity:
        os.sched_setaffinity(0, set(affinity))
    policy = OnboardPolicy(
        robot_id=args.robot, **files, device=args.device or raw.get("device", "cpu"),
        inference_precision=raw.get("inference_precision", "fp32"),
        control_mode=int(raw["control_mode"]), future_step=int(raw["future_step"]),
        residual_scale=float(raw["residual_scale"]), start_frame=int(raw["start_frame"]),
        reference_alignment="motion_world", torch_num_threads=int(
            config.get("acceleration", {}).get("torch_num_threads", raw.get("torch_num_threads", 4))),
        anchor_angular_velocity_frame=contract["anchor_angular_velocity_frame"],
    )
    default = load_default_command(standing_dir)
    standing = OnboardStanding(policy, default)
    acceleration = config.get("acceleration", {})
    accelerated_backend = None
    if acceleration.get("backend", "pytorch") == "tensorrt":
        from dual_runtime.tensorrt_backend import attach_tensorrt
        accelerated_backend = attach_tensorrt(policy,
            resolve(path.parent, acceleration["directory"]), files, args.robot,
            python=acceleration.get("python", "/usr/bin/python3"))
    elif acceleration.get("backend", "pytorch") != "pytorch":
        raise ValueError("unknown acceleration backend")
    try:
        if accelerated_backend is not None:
            accelerated_backend.timeout_s = 5.
        policy.warmup()
        if accelerated_backend is not None:
            accelerated_backend.timeout_s = .08
    except BaseException:
        if accelerated_backend is not None:
            accelerated_backend.close()
        raise
    limits = load_yaml(limits_path)
    local = net[args.robot]
    udp = dict(local["bridge_udp"])
    for key in ("state_bind_host", "cmd_host"):
        if not ipaddress.ip_address(udp[key]).is_loopback:
            raise ValueError("onboard bridge must use loopback")
    team = TeamState(args.robot, identity, start_delay_s=safety["start_delay_s"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    directory = Path(args.log_dir) / (time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}_{args.robot}")
    directory.mkdir(parents=True)
    rows = []
    robot = pose_channel = peer_channel = None
    reason = "duration_reached"
    frame = -1
    ready = False
    latest_snapshot_seq = -1
    start = time.monotonic()
    started_control = False
    previous_buttons = None
    default_start = None
    default_ticks = round(control["default_pose_duration_s"] * 50)
    elapsed_default = 0
    completed = False
    previous_reference = None
    peer_seen = False
    slow_ticks = 0
    last_send = None
    last_report = 0.
    try:
        pose_channel = LatestChannel(*channel_endpoints(net, args.robot, "pose"),
            "pose", max_rtt_s=net["max_clock_rtt_s"])
        peer_channel = LatestChannel(*channel_endpoints(net, args.robot, "team"),
            "team", max_rtt_s=net["max_clock_rtt_s"])
        robot = RobotSession(RobotSessionConfig(
            robot_id=args.robot, udp=udp,
            lower=np.asarray(limits["joint_pos_lowerlimit_lab"]),
            upper=np.asarray(limits["joint_pos_upperlimit_lab"]),
            kp=policy.kp, kd=policy.kd, torque_limit=policy.torque_limit,
            damping_kd=control["damping_kd"],
            max_target_delta=control["phase_target_delta"]["default_pose"],
        ), client=MotionBridgeClient(udp, require_bridge_session=True))
        print(f"Onboard robot={args.robot} actuation={args.actuate} identity={identity}\n"
              f"Inference backend: {'TensorRT/CUDA (local worker)' if accelerated_backend is not None else 'PyTorch'}\n"
              "Start/s → DefaultPose; B/b → standing; A/a → task; Stop/x → damping.\n"
              "Terminal keys require Enter; either robot remote can request a transition.", flush=True)
        deadline = time.monotonic()
        while args.duration is None or time.monotonic()-start < args.duration:
            tick_started = time.monotonic()
            step = None
            peer_channel.publish(team.status(ready=ready, frame=frame))
            # A short bounded local poll; wireless reads never block this loop.
            state = robot.read(.001)
            if state is None:
                state = robot.last_state
            try:
                if state is None or (time.monotonic_ns()-state.packet_arrival_ns)*1.e-9 > safety["local_state_timeout_s"]:
                    raise RuntimeError("local_bridge_state_stale")
                payload, offset, uncertainty = pose_channel.read(safety["pose_timeout_s"])
                snapshot = decode_pose(payload, calibration_id=calibration_id,
                    offset=offset, uncertainty=uncertainty, max_age_s=safety["pose_timeout_s"])
                latest_snapshot_seq = int(payload["snapshot_seq"])
                peer, peer_offset, peer_uncertainty = peer_channel.read(safety["peer_timeout_s"])
            except RuntimeError as exc:
                ready = False
                if started_control or peer_seen or time.monotonic()-start > safety["startup_timeout_s"]:
                    raise
                # Handshake/startup stays in zero torque, including after a
                # new local bridge session is acknowledged.
                if state is not None:
                    zero = np.zeros(29, dtype=np.float32)
                    robot.send_pd(PDCommand(zero, zero, zero), state, enable=0)
                previous_buttons = None
                if time.monotonic()-last_report > 2:
                    print(f"WAITING: {exc}", flush=True)
                    last_report = time.monotonic()
                time.sleep(.01)
                deadline = time.monotonic()
                continue
            peer_seen = True
            leader_now = time.time() + (peer_offset if args.robot == "b" else 0.)
            ready = team.phase in {"zero", "standing", "finished"} or (
                team.phase == "default" and elapsed_default >= default_ticks
            ) or (team.phase == "executing" and completed)
            buttons = {key: bool(state.buttons.get(key, False)) for key in ("start", "B", "A", "stop")}
            if buttons["stop"]:
                team.request("stop")
            if previous_buttons is not None:
                for button, event in (("start", "start"), ("B", "standing"), ("A", "task")):
                    if buttons[button] and not previous_buttons[button] and ready:
                        team.request(event)
            previous_buttons = buttons
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip().lower()
                if key in {"s", "b", "a", "x"}:
                    team.request(dict(s="start", b="standing", a="task", x="stop")[key])
            transitioned = team.update(peer, leader_now, ready=ready, complete=completed)
            if transitioned:
                print(f"robot={args.robot} epoch={team.epoch} phase={team.phase}", flush=True)
                started_control = True
                if team.phase == "default":
                    default_start = state.q_lab.copy()
                    robot.prepare(state)
                elif team.phase in {"standing", "finished"}:
                    standing.reset(snapshot)
                elif team.phase == "executing":
                    pre = raw.get("preflight", {})
                    policy.initialize(snapshot, **{key: float(pre.get(key, value)) for key, value in {
                        "max_partner_position_error_m": .2, "max_object_position_error_m": .2,
                        "max_box_size_error_m": .03, "max_robot_orientation_error_rad": .35,
                        "max_object_orientation_error_rad": .35}.items()})
                    policy.reset_rollout()
            if team.phase == "zero":
                zero = np.zeros(29, dtype=np.float32)
                desired = PDCommand(zero, zero, zero)
            elif team.phase == "default":
                elapsed_default += 1
                alpha = min(1., elapsed_default/default_ticks)
                desired = PDCommand(default_start*(1-alpha)+default.target_pos*alpha, default.kp, default.kd)
            elif team.phase in {"standing", "finished"}:
                q = state.quat_wxyz
                tilt = np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1))
                if tilt > control["max_tilt_rad"]:
                    raise RuntimeError("standing_tilt_limit")
                desired = standing.compute(state, snapshot)
            else:
                if peer["phase"] == "executing" and frame >= 0 and int(peer["frame"]) >= 0:
                    if abs(frame-int(peer["frame"])) > safety["max_frame_skew"]:
                        raise RuntimeError("peer_reference_frame_skew")
                if not completed:
                    expected = policy.start_frame + int(max(0, leader_now-team.started_at)*50)
                    if abs(policy.frame-expected) > safety["max_frame_skew"]:
                        raise RuntimeError("local_reference_deadline_missed")
                    step = policy.compute_single(state, snapshot)
                    frame, completed = step.frame, step.complete
                    desired = PDCommand(step.target, policy.kp, policy.kd)
                else:
                    desired = standing.compute(state, snapshot)
                reference = policy.reference.frame(frame)[0].object_pos_w
                delta = snapshot.object.position_w - (reference if previous_reference is None else previous_reference)
                if abs(delta[2]) > control["task_safety"]["object_position_z_error_m"] or np.linalg.norm(delta) > control["task_safety"]["object_position_xyz_error_m"]:
                    raise RuntimeError("object_reference_error")
                previous_reference = reference.copy()
                if step is not None and step.complete:
                    standing.reset(snapshot)
            phase_key = {"default": "default_pose", "standing": "scalebfm_standing", "finished": "scalebfm_standing"}.get(team.phase, "executing")
            robot.limiter.max_target_delta = control["phase_target_delta"][phase_key]
            processing = time.monotonic()-tick_started
            slow_ticks = slow_ticks+1 if processing > safety["max_processing_s"] else 0
            if slow_ticks >= safety["max_slow_ticks"]:
                raise RuntimeError("onboard_processing_budget_exceeded")
            if processing > safety["max_command_gap_s"]:
                raise RuntimeError("onboard_control_stall")
            if last_send is not None and time.monotonic()-last_send > safety["max_command_gap_s"]:
                raise RuntimeError("onboard_command_gap")
            command = robot.send_pd(desired, state, enable=int(args.actuate and team.phase != "zero"))
            last_send = time.monotonic()
            ready = team.phase in {"zero", "standing", "finished"} or (team.phase == "default" and elapsed_default >= default_ticks) or (team.phase == "executing" and completed)
            peer_channel.publish(team.status(ready=ready, frame=frame))
            rows.append(dict(time_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                phase=team.phase, epoch=team.epoch, phase_started_at=team.started_at,
                frame=frame, pose_seq=latest_snapshot_seq, state_arrival_ns=state.packet_arrival_ns,
                q=state.q_lab.copy(), dq=state.dq_lab.copy(), target=command.target_pos.copy(),
                kp=command.kp.copy(), kd=command.kd.copy(), imu_quat_wxyz=state.quat_wxyz.copy(),
                gyro=state.gyro.copy(),
                poses=np.asarray(payload["poses"], dtype=np.float32),
                enable=robot.last_enable, processing_s=processing, peer_frame=int(peer["frame"]),
                peer_clock_offset_s=peer_offset, peer_clock_uncertainty_s=peer_uncertainty,
                observation=np.full(201, np.nan) if step is None else step.observation,
                residual=np.full(29, np.nan) if step is None else step.residual))
            if time.monotonic()-last_report > 2:
                print(f"robot={args.robot} phase={team.phase} frame={frame} processing_ms={processing*1000:.2f}", flush=True)
                last_report = time.monotonic()
            deadline += .02
            delay = deadline-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif -delay > safety["max_command_gap_s"]:
                raise RuntimeError("control_scheduler_stall")
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        team.fault = reason
        # Publish fault before teardown; absence is also detected by peer lease.
        if peer_channel is not None:
            for _ in range(3):
                try:
                    peer_channel.publish(team.status(ready=False, frame=frame))
                except OSError:
                    break
        try:
            if robot is not None and robot.last_state is not None:
                robot.send_hold(robot.last_state)
        finally:
            if robot is not None:
                robot.close()
            if accelerated_backend is not None:
                accelerated_backend.close()
            for channel in (pose_channel, peer_channel):
                if channel is not None:
                    channel.close()
            np.savez_compressed(directory / "rollout.npz", **{
                key: np.asarray([r[key] for r in rows]) for key in (rows[0] if rows else {})})
            (directory / "metadata.json").write_text(json.dumps(dict(
                robot=args.robot, identity=identity, exit_reason=reason, contract=contract,
                calibration_id=calibration_id, config=config, policy_config=raw,
                run_id=None if peer_channel is None else (
                    peer_channel.stream if args.robot == "a" else peer_channel.remote_stream),
                pose_stream=None if pose_channel is None else pose_channel.remote_stream,
                actuation=args.actuate, ticks=len(rows)), indent=2))
            print(f"saved onboard log: {directory}", flush=True)


if __name__ == "__main__":
    main()
