#!/usr/bin/env python3
"""Minimal dual-G1 independent-routing verification entry point.

This is deliberately separate from the eventual Dual ScaleBFM policy entry
point.  It proves that two bridge UDP endpoints and two DDS interfaces are
isolated before enabling a learned policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from dual_runtime import (
    IndependentDualCoordinator,
    IndependentTargetPolicy,
    RobotSession,
    RobotSessionConfig,
)


ACTUATION_CONFIRMATION = "ENABLE_MOTORS"
ROOT = Path(__file__).resolve().parents[1]


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def _session(raw: dict[str, Any], robot_id: str) -> RobotSession:
    controller_path = Path(raw.get("controller_config", ROOT / "config/g1/controller.yaml"))
    if not controller_path.is_absolute():
        controller_path = ROOT / "config/g1" / controller_path
    controller = _load(controller_path)
    return RobotSession(
        RobotSessionConfig(
            robot_id=robot_id,
            udp=dict(raw["udp"]),
            lower=np.asarray(raw.get("joint_lower", [-3.0] * 29), dtype=np.float32),
            upper=np.asarray(raw.get("joint_upper", [3.0] * 29), dtype=np.float32),
            kp=np.asarray(raw.get("kp", controller["kps"]), dtype=np.float32),
            kd=np.asarray(raw.get("kd", controller["kds"]), dtype=np.float32),
            max_target_delta=float(raw.get("max_target_delta", 0.02)),
            damping_kd=float(raw.get("damping_kd", 8.0)),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/g1/dual_omnicontact.yaml"))
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--state-timeout", type=float, default=None)
    parser.add_argument(
        "--act-robot",
        choices=("none", "a", "b", "both"),
        default="none",
        help="select which robot receives enabled PD targets",
    )
    parser.add_argument(
        "--act",
        action="store_true",
        help="deprecated shorthand for --act-robot both",
    )
    parser.add_argument("--confirm-actuation", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = _load(args.config)
    if args.act and args.act_robot != "none":
        raise SystemExit("use either --act or --act-robot, not both")
    act_robot = "both" if args.act else args.act_robot
    if act_robot != "none" and args.confirm_actuation != ACTUATION_CONFIRMATION:
        raise SystemExit(f"motor output requires --confirm-actuation {ACTUATION_CONFIRMATION}")
    a_raw = config["robot_a"]
    b_raw = config["robot_b"]
    session_a = _session(a_raw, "a")
    session_b = _session(b_raw, "b")
    coordinator = IndependentDualCoordinator(
        session_a,
        session_b,
        IndependentTargetPolicy(int(a_raw["test_joint_index"]), float(a_raw["test_offset_rad"])),
        IndependentTargetPolicy(int(b_raw["test_joint_index"]), float(b_raw["test_offset_rad"])),
        state_timeout_s=float(args.state_timeout or config.get("state_timeout_s", 0.2)),
        max_state_skew_s=float(config.get("max_state_skew_s", 0.05)),
        fail_closed=bool(config.get("fail_closed", True)),
    )
    duration = float(
        config.get("duration_s", 10.0) if args.duration is None else args.duration
    )
    if duration <= 0.0:
        raise SystemExit("duration must be positive")
    enable_a = int(act_robot in {"a", "both"})
    enable_b = int(act_robot in {"b", "both"})
    print(
        f"dual verification: duration={duration:.1f}s act_robot={act_robot} "
        f"config={args.config}"
    )
    start = time.monotonic()
    ticks = 0
    try:
        while time.monotonic() - start < duration:
            ok, reason = coordinator.step(enable_a=enable_a, enable_b=enable_b)
            if not ok:
                raise RuntimeError(reason)
            ticks += 1
            if ticks % 50 == 0:
                print(f"ticks={ticks} A_joint={a_raw['test_joint_index']} B_joint={b_raw['test_joint_index']}")
    finally:
        coordinator.close()
    print(f"dual verification complete: ticks={ticks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
