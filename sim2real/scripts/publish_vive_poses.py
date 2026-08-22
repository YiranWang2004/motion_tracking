#!/usr/bin/env python3
"""Read two local Vive Trackers and publish calibrated poses over wired UDP."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnicontact.perception.pose_udp import PoseUdpPublisher
from omnicontact.perception.vive_pose import ViveDeploymentConfig, VivePoseProvider


logger = logging.getLogger("omnicontact.vive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish calibrated OmniContact Vive poses to the robot policy host"
    )
    parser.add_argument("--vive-config", required=True)
    parser.add_argument("--target-ip", required=True, help="IP of the onboard pose receiver")
    parser.add_argument("--port", type=int, default=15150)
    parser.add_argument("--token", default=os.environ.get("OMNICONTACT_POSE_TOKEN"))
    parser.add_argument("--bind-ip", default=None, help="optional workstation Ethernet IP")
    parser.add_argument("--vive-hz", type=float, default=100.0)
    parser.add_argument("--send-hz", type=float, default=100.0)
    parser.add_argument("--wait-timeout", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.token:
        raise SystemExit("--token or OMNICONTACT_POSE_TOKEN is required")
    if args.vive_hz <= 0.0 or args.send_hz <= 0.0:
        raise SystemExit("--vive-hz and --send-hz must be positive")

    config = ViveDeploymentConfig.load(args.vive_config)
    provider = VivePoseProvider(config, poll_hz=args.vive_hz)
    publisher = PoseUdpPublisher(
        args.target_ip,
        args.port,
        args.token,
        bind_ip=args.bind_ip,
    )
    sent_packets = 0
    last_pose_stamp = -1.0
    last_status = time.monotonic()
    interval = 1.0 / args.send_hz

    try:
        devices = provider.start()
        logger.info("OpenVR Trackers: %s", devices)
        if not provider.wait_until_ready(args.wait_timeout):
            raise RuntimeError("timed out waiting for both calibrated Tracker poses")
        logger.warning(
            "Publishing calibrated poses to %s:%d (stream %s)",
            args.target_ip,
            args.port,
            publisher.stream_id,
        )
        next_tick = time.monotonic()
        while True:
            if provider.error is not None:
                raise RuntimeError(
                    f"Vive pose provider failed: {provider.error}"
                ) from provider.error
            robot_pose, object_pose = provider.get_poses()
            if (
                robot_pose is not None
                and object_pose is not None
                and object_pose.stamp_s > last_pose_stamp
            ):
                publisher.send(robot_pose, object_pose)
                last_pose_stamp = object_pose.stamp_s
                sent_packets += 1

            now = time.monotonic()
            if now - last_status >= 1.0:
                logger.info(
                    "UDP sent=%d, Vive valid=%d invalid=%d",
                    sent_packets,
                    provider.valid_updates,
                    provider.invalid_updates,
                )
                last_status = now
            next_tick += interval
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -interval:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logger.warning("Publisher interrupted")
    finally:
        provider.stop()
        publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
