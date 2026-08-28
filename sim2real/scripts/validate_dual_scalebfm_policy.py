#!/usr/bin/env python3
"""Offline artifact/shape/FK/numerics/latency validation without bridge output."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deploy_dual_scalebfm_residual import resolve_artifacts
from dual_runtime.dual_pose_provider import DualPoseSnapshot
from dual_runtime.scalebfm_residual_policy import DualScaleBFMResidualPolicy
from omnicontact.contracts import ObjectPose, RobotPose
from omnicontact.runtime import BridgeState

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config/g1/dual_scalebfm_residual.yaml")
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 10:
        raise SystemExit("--steps must be at least 10")
    config_path = Path(args.config).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = resolve_artifacts(config_path, raw)
    device = args.device or raw.get("device", "cpu")
    policy = DualScaleBFMResidualPolicy(
        scalebfm_checkpoint=artifacts["scalebfm_checkpoint"],
        scalebfm_metadata=artifacts["scalebfm_metadata"],
        scalebfm_mode_table=artifacts["scalebfm_mode_table"],
        residual_checkpoint=artifacts["residual_checkpoint"],
        reference_bundle=artifacts["reference_bundle"],
        kinematics_xml=artifacts["kinematics_xml"],
        device=device,
        inference_precision=raw.get("inference_precision", "fp32"),
        control_mode=int(raw.get("control_mode", 7)),
        future_step=int(raw.get("future_step", 5)),
        residual_scale=float(raw.get("residual_scale", 0.1)),
        start_frame=int(raw.get("start_frame", 1)),
        reference_alignment="none",
        torch_num_threads=int(raw.get("torch_num_threads", 4)),
    )
    references = policy.reference.frame(policy.start_frame)
    stamp = time.monotonic()
    poses = tuple(
        RobotPose(
            reference.body_pos_w[0],
            reference.body_quat_wxyz[0][[1, 2, 3, 0]],
            stamp,
        )
        for reference in references
    )
    snapshot = DualPoseSnapshot(
        poses[0],
        poses[1],
        ObjectPose(
            references[0].object_pos_w,
            references[0].object_quat_wxyz[[1, 2, 3, 0]],
            policy.reference.box_half_extents,
            stamp,
        ),
    )
    policy.initialize(
        snapshot,
        max_partner_position_error_m=1.0e-4,
        max_object_position_error_m=1.0e-4,
        max_box_size_error_m=1.0e-4,
        max_robot_orientation_error_rad=1.0e-4,
        max_object_orientation_error_rad=1.0e-4,
    )
    elapsed = []
    max_residual = 0.0
    for sequence in range(args.steps):
        references = policy.reference.frame(policy.frame)
        arrival = time.monotonic_ns()
        states = tuple(
            BridgeState(
                q_lab=reference.joint_pos,
                dq_lab=reference.joint_vel,
                quat_wxyz=reference.body_quat_wxyz[0],
                gyro=np.zeros(3, dtype=np.float32),
                buttons={},
                state_receive_time_ns=None,
                packet_seq=sequence,
                packet_arrival_ns=arrival,
            )
            for reference in references
        )
        started = time.perf_counter()
        step = policy.compute(states, snapshot)
        elapsed.append(time.perf_counter() - started)
        if not np.all(np.isfinite(step.targets)):
            raise RuntimeError("offline policy produced a non-finite target")
        max_residual = max(max_residual, float(np.max(np.abs(step.residuals))))
    timings = 1000.0 * np.asarray(elapsed[5:])
    print(f"device={device} steps={args.steps}")
    print(
        f"latency_ms mean={timings.mean():.3f} p95={np.percentile(timings, 95):.3f} "
        f"max={timings.max():.3f}"
    )
    print(f"max_abs_residual={max_residual:.6f}")
    budget_ms = 1000.0 * float(raw.get("max_inference_time_s", 0.018))
    if np.percentile(timings, 95) > budget_ms:
        raise RuntimeError(
            f"P95 inference latency exceeds configured {budget_ms:.1f} ms budget"
        )
    print("offline dual policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
