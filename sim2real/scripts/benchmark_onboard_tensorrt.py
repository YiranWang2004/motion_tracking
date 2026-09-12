#!/usr/bin/env python3
"""Offline end-to-end checks using reference poses; never creates a bridge."""

from omnicontact.runtime import BridgeState
from omnicontact.contracts import RobotPose, ObjectPose
from dual_runtime.dual_pose_provider import DualPoseSnapshot
from dual_runtime.interactive_control import load_default_command
from dual_runtime.tensorrt_backend import attach_tensorrt
from dual_runtime.onboard_policy import OnboardPolicy, OnboardStanding
from dual_runtime.onboard_config import (
    load_onboard_config,
    load_yaml,
    load_artifacts,
    resolve,
)
import argparse
import json
import sys
import time
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def stats(values):
    return dict(
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        max_ms=float(max(values)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--engines", required=True)
    p.add_argument("--robot", required=True, choices=["a", "b"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output")
    a = p.parse_args()
    path = Path(a.config).resolve()
    config = load_onboard_config(path)
    raw = load_yaml(resolve(path.parent, config["policy_config"]))
    files, contract = load_artifacts(resolve(path.parent, config["artifact_directory"]))
    policy = OnboardPolicy(
        robot_id=a.robot,
        **files,
        device="cpu",
        inference_precision="fp32",
        control_mode=raw["control_mode"],
        future_step=raw["future_step"],
        residual_scale=raw["residual_scale"],
        start_frame=raw["start_frame"],
        reference_alignment="motion_world",
        torch_num_threads=1,
        anchor_angular_velocity_frame=contract["anchor_angular_velocity_frame"],
    )
    refs = policy.reference.frame(policy.start_frame)
    poses = [
        RobotPose(r.body_pos_w[0], r.body_quat_wxyz[0][[1, 2, 3, 0]], time.monotonic())
        for r in refs
    ]
    snap = DualPoseSnapshot(
        *poses,
        ObjectPose(
            refs[0].object_pos_w,
            refs[0].object_quat_wxyz[[1, 2, 3, 0]],
            policy.reference.box_half_extents,
            time.monotonic(),
        ),
    )
    r = refs[policy.index]
    state = BridgeState(
        r.joint_pos,
        r.joint_vel,
        r.body_quat_wxyz[0],
        np.zeros(3),
        {},
        None,
        0,
        time.monotonic_ns(),
    )

    def run():
        policy.warmup()
        policy.reset_rollout()
        policy.initialized = True
        output = []
        timings = []
        for _ in range(a.steps):
            start = time.perf_counter()
            step = policy.compute_single(state, snap)
            timings.append((time.perf_counter() - start) * 1000)
            output.append(step.target.copy())
        return np.stack(output), timings

    cpu, cput = run()
    print("CPU", stats(cput), flush=True)
    backend = attach_tensorrt(policy, a.engines, files, a.robot, timeout_s=5.0)
    try:
        gpu, gput = run()
        backend.timeout_s = 0.08
        np.testing.assert_allclose(gpu, cpu, atol=0.002, rtol=0.001)
        standing = OnboardStanding(
            policy, load_default_command(resolve(path.parent, "omnicontact"))
        )
        standing.reset(snap)
        for _ in range(10):
            standing.compute(state, snap)
        standing_times = []
        for _ in range(a.steps):
            start = time.perf_counter()
            standing.compute(state, snap)
            standing_times.append((time.perf_counter() - start) * 1000)
        report = dict(
            robot=a.robot,
            task=str(path),
            engine_runtime=backend.info["tensorrt"],
            steps=a.steps,
            cpu_task=stats(cput),
            tensorrt_task=stats(gput),
            tensorrt_standing=stats(standing_times),
            max_target_error_rad=float(np.abs(gpu - cpu).max()),
            meets_18ms_max=max(gput + standing_times) < 18,
        )
        print(json.dumps(report, indent=2), flush=True)
        if a.output:
            Path(a.output).write_text(json.dumps(report, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
