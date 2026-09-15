"""Single G1 bridge controller, identical on host and onboard computers."""
# ruff: noqa: I001

import argparse
import json
import time
from pathlib import Path

# Load PyTorch/OpenMP first: older Jetson glibc can exhaust static TLS otherwise.
import torch  # noqa: F401
import mujoco
import numpy as np
from omnicontact.contracts import PDCommand, RobotPose
from omnicontact.runtime import CommandLimiter, MotionBridgeClient
from scalebfm.constants import POLICY_JOINT_NAMES
from scalebfm.kinematics import G1PolicyKinematics
from scalebfm.policy import ScaleBFMHistory

from .config import load_config, load_policy
from .pose import LocalOdometry, ViveSource
from .reference import Frame, PicoSource, ReferenceWindow, rotation


def standing_window(fk, joints, pose):
    live = fk.forward(joints, pose)
    return (
        np.repeat(live.body_pos_w[None, None], 6, axis=1),
        np.repeat(live.body_quat_wxyz[None, None], 6, axis=1),
    )


def limit_command(target, state, policy, limiter):
    """Intersect joint, slew and estimated PD-torque constraints; fail if empty."""
    lower = np.maximum(limiter.lower, limiter.last_target - limiter.max_target_delta)
    upper = np.minimum(limiter.upper, limiter.last_target + limiter.max_target_delta)
    if np.any(policy.kp <= 0):
        raise ValueError("ScaleBFM stiffness must be positive")
    lower = np.maximum(
        lower,
        state.q_lab + (policy.kd * state.dq_lab - policy.torque_limit) / policy.kp,
    )
    upper = np.minimum(
        upper,
        state.q_lab + (policy.kd * state.dq_lab + policy.torque_limit) / policy.kp,
    )
    if np.any(lower > upper):
        raise RuntimeError("no target satisfies joint/slew/torque limits")
    return limiter.apply(PDCommand(np.clip(target, lower, upper), policy.kp, policy.kd))


def benchmark(policy, fk, cfg, count):
    """No sockets, no robot: exercise FK/history/task construction and inference."""
    pose = RobotPose(np.array([0, 0, 0.78]), np.array([0, 0, 0, 1]), 0.0)
    frames = [
        Frame(i * 0.02, pose.position_w, np.array([1, 0, 0, 0]), policy.default_q)
        for i in range(35)
    ]
    window = ReferenceWindow(fk, cfg["time_offsets"])
    window.reset(frames[0], pose, policy.default_q, 0.0)
    history = ScaleBFMHistory()
    action = np.zeros(29)
    samples = []
    for i in range(count + 5):
        started = time.monotonic()
        ref = window.build(frames, 1.0)
        live = fk.forward(policy.default_q, pose)
        history.update(
            np.array([1, 0, 0, 0]), np.zeros(3), policy.default_q, np.zeros(29), action
        )
        target, action_batch = policy.infer_batch(
            (history,),
            *ref,
            live.body_pos_w[None],
            live.body_quat_wxyz[None],
            control_mode=cfg["control_mode"],
            time_offsets=np.array(cfg["time_offsets"]),
        )
        action = action_batch[0]
        if i >= 5:
            samples.append((time.monotonic() - started) * 1000)
        assert target.shape == (1, 29) and np.isfinite(target).all()
    report = {
        "samples": count,
        "pipeline_ms_p50": float(np.percentile(samples, 50)),
        "pipeline_ms_p95": float(np.percentile(samples, 95)),
        "max_ms": max(samples),
        "finite": True,
        "hardware_tested": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def run(cfg, policy, fk, actuate=False, duration=None):
    model = fk.kinematics.model
    indices = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in POLICY_JOINT_NAMES
    ]
    limiter = CommandLimiter(
        model.jnt_range[indices, 0],
        model.jnt_range[indices, 1],
        cfg["max_target_delta"],
    )
    source = PicoSource(cfg["pico_host"], cfg["source_timeout_s"])
    pose_source = None
    client = None
    active = False
    state = None
    try:
        pose_source = (
            LocalOdometry(fk, **cfg.get("local_odometry", {}))
            if cfg["pose_mode"] == "local"
            else ViveSource(cfg["vive_endpoint"])
        )
        client = MotionBridgeClient(cfg["udp"], require_bridge_session=actuate)
        # Observe-only must not negotiate a command session on session-aware bridges.
        if not actuate:
            client.bridge_session = None
        reference = ReferenceWindow(fk, cfg["time_offsets"])
        history = ScaleBFMHistory()
        previous_action = np.zeros(29)
        phase = "waiting"
        generation = None
        tracking = False
        started = time.monotonic()
        next_tick = started
        last_log = 0.0
        print(
            "ACTUATION ARMED" if actuate else "OBSERVE ONLY: no commands sent",
            flush=True,
        )
        print(
            "G1 Start: interpolate to default; G1 A: policy; Pico A: follow/re-anchor; Pico X: freeze; G1 Select: exit",
            flush=True,
        )
        while duration is None or time.monotonic() - started < duration:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick = max(next_tick + 0.02, time.monotonic())
            state = client.read_next(cfg["state_timeout_s"])
            now = time.monotonic()
            source.poll(now)
            if state is None:
                if active:
                    raise RuntimeError("bridge state timeout")
                continue
            if now - state.packet_arrival_ns * 1e-9 > cfg["state_timeout_s"]:
                raise RuntimeError("stale bridge state")
            if state.buttons["stop"]:
                break
            try:
                pose = pose_source.update(state, now)
            except RuntimeError as exc:
                if active:
                    raise
                if now - last_log > 2:
                    print(f"Waiting for pose: {exc}", flush=True)
                    last_log = now
                continue
            if rotation(state.quat_wxyz).as_matrix()[2, 2] < np.cos(
                cfg["max_tilt_rad"]
            ):
                raise RuntimeError("pelvis tilt limit exceeded")
            if phase == "waiting":
                if not actuate or client.button_rise["start"]:
                    phase, stand_start = "stand", now
                    start_q = state.q_lab.copy()
                    limiter.reset(start_q)
                    active = actuate
                    print("Interpolating to ScaleBFM default posture", flush=True)
                else:
                    continue
            compute_start = time.monotonic()
            if phase == "stand":
                alpha = np.clip((now - stand_start) / cfg["stand_s"], 0, 1)
                target = start_q + alpha * (policy.default_q - start_q)
                if alpha == 1 and (not actuate or client.button_rise["A"]):
                    phase = "policy"
                    history.reset()
                    held = standing_window(fk, policy.default_q, pose)
                    source.enabled = False  # Require a NEW Pico press after G1 A.
                    print("ScaleBFM active; waiting for Pico A", flush=True)
            else:
                if source.enabled:
                    if not source.fresh(now):
                        raise RuntimeError("Pico reference stale")
                    if generation != source.generation:
                        reference.reset(source.frames[-1], pose, state.q_lab, now)
                        generation = source.generation
                    held = reference.build(source.frames, now)
                    tracking = True
                elif tracking:
                    # Freeze at the most recent target, not a six-frame moving loop.
                    held = tuple(np.repeat(x[:, :1], 6, axis=1) for x in held)
                    tracking = False
                live = fk.forward(state.q_lab, pose)
                history.update(
                    pose.quaternion_xyzw[[3, 0, 1, 2]],
                    state.gyro,
                    state.q_lab,
                    state.dq_lab,
                    previous_action,
                )
                targets, _ = policy.infer_batch(
                    (history,),
                    *held,
                    live.body_pos_w[None],
                    live.body_quat_wxyz[None],
                    control_mode=cfg["control_mode"],
                    time_offsets=np.asarray(cfg["time_offsets"]),
                )
                target = targets[0]
            command = limit_command(target, state, policy, limiter)
            finished = time.monotonic()
            if (
                finished - compute_start > cfg["compute_timeout_s"]
                or finished - state.packet_arrival_ns * 1e-9 > cfg["state_timeout_s"]
            ):
                raise RuntimeError("control deadline missed; target discarded")
            if phase == "policy" and source.enabled and not source.fresh(finished):
                raise RuntimeError("Pico expired during inference; target discarded")
            if cfg["pose_mode"] == "vive" and finished - pose.stamp_s > 0.1:
                raise RuntimeError("Vive expired during inference; target discarded")
            previous_action = (
                command.target_pos - policy.default_q
            ) / policy.action_scale
            if actuate:
                client.send(command, enable=1, state=state)
            if now - last_log > 2:
                print(
                    f"phase={phase} pose={cfg['pose_mode']} reference={'live' if tracking else 'held'} compute_ms={(finished - compute_start) * 1000:.1f}",
                    flush=True,
                )
                last_log = now
    finally:
        try:
            if client is not None:
                try:
                    if active and client.latest_state is not None:
                        client.send_damping(client.latest_state)
                except Exception as exc:  # noqa: BLE001 - preserve original fault during cleanup
                    print(
                        f"Damping send failed; bridge watchdog must stop control: {exc}",
                        flush=True,
                    )
                finally:
                    client.close()
        finally:
            try:
                source.close()
            finally:
                if pose_source is not None:
                    pose_source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2] / "config/g1/tracking_scalebfm.yaml"
        ),
    )
    parser.add_argument("--pose-mode", choices=("local", "vive"))
    parser.add_argument("--pico-host")
    parser.add_argument("--cmd-host")
    parser.add_argument("--vive-endpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--check",
        action="store_true",
        help="offline benchmark only; opens no robot/Pico sockets",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--trt-directory")
    parser.add_argument("--trt-python", default="/usr/bin/python3")
    args = parser.parse_args()
    cfg, files = load_config(args.config)
    for key in ("pose_mode", "pico_host", "vive_endpoint"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.cmd_host:
        cfg["udp"]["cmd_host"] = args.cmd_host
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.duration is not None and (
        not np.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("duration must be finite and positive")
    policy = load_policy(files, args.device)
    fk = G1PolicyKinematics(files["xml"])
    backend = None
    try:
        if args.trt_directory:
            from scalebfm.tensorrt_backend import TensorRTBackend

            if args.device != "cpu":
                parser.error("TensorRT worker requires --device cpu for preprocessing")
            backend = TensorRTBackend(
                args.trt_directory,
                files,
                side=None,
                python=args.trt_python,
                timeout_s=cfg["compute_timeout_s"],
            )
            policy.accelerated_inference = backend.scale
        result = benchmark(policy, fk, cfg, args.samples)
        if not args.check:
            if args.actuate and result["pipeline_ms_p95"] >= 20:
                raise RuntimeError(
                    "50Hz preflight failed: p95 >= 20ms; use a faster backend"
                )
            run(cfg, policy, fk, args.actuate, args.duration)
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
