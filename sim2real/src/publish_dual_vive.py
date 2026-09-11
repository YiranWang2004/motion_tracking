#!/usr/bin/env python3
"""Vive host: calibrate three trackers and distribute poses only."""
import argparse
import logging
from pathlib import Path
import time

from dual_runtime.onboard_config import load_onboard_config, channel_endpoints, resolve, sha256
from dual_runtime.onboard_network import LatestChannel, pose_payload
from dual_runtime.vive_dual_pose import DualViveDeploymentConfig, DualVivePoseProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/g1/onboard_scalebfm.yaml")
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    path = Path(args.config).expanduser().resolve()
    config = load_onboard_config(path)
    network = config["network"]
    calibration = resolve(path.parent, config["vive_config"])
    calibration_id = sha256(calibration)
    logging.basicConfig(level=logging.INFO)
    provider = DualVivePoseProvider(DualViveDeploymentConfig.load(calibration))
    channels = []
    print(f"Vive calibration SHA256: {calibration_id}", flush=True)
    try:
        for side in ("a", "b"):
            channels.append(LatestChannel(
                *channel_endpoints(network, side, "publisher"), "pose",
                max_rtt_s=float(network["max_clock_rtt_s"]),
            ))
        # One host acquisition stream shared by both destinations.
        channels[1].stream = channels[0].stream
        provider.start()
        if not provider.wait_until_ready(10.):
            raise RuntimeError("Vive startup timeout")
        start = time.monotonic()
        sample_seq = 0
        last_stamp = None
        while args.duration is None or time.monotonic() - start < args.duration:
            if provider.error:
                raise RuntimeError("Vive acquisition failed") from provider.error
            snapshot = provider.get_snapshot()
            if snapshot is not None:
                if snapshot.robot_a.stamp_s != last_stamp:
                    sample_seq += 1
                    last_stamp = snapshot.robot_a.stamp_s
                payload = pose_payload(snapshot, calibration_id, sample_seq)
                for channel in channels:
                    channel.publish(payload)
            time.sleep(.01)
    finally:
        provider.stop()
        for channel in channels:
            channel.close()


if __name__ == "__main__":
    main()
