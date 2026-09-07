#!/usr/bin/env python3
"""List all OpenVR GenericTrackers and print a short pose health check."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnicontact.perception.openvr_tracker import OpenVRTrackerReader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        raise SystemExit("--seconds must be positive")

    reader = OpenVRTrackerReader()
    try:
        devices = reader.start()
        print("Detected GenericTrackers:")
        for serial, index in sorted(devices.items()):
            print(f"  serial={serial}  openvr_index={index}")
        valid_counts = {serial: 0 for serial in devices}
        missing_frames = {serial: [] for serial in devices}
        consecutive_missing = {serial: 0 for serial in devices}
        longest_missing = {serial: 0 for serial in devices}
        attempts = 0
        start = time.monotonic()
        end = start + args.seconds
        print("Frame numbers are 1-based read attempts, not SteamVR frame IDs.", flush=True)
        while time.monotonic() < end:
            samples = reader.read_all()
            attempts += 1
            elapsed = time.monotonic() - start
            missing = []
            for serial in devices:
                if samples.get(serial) is not None:
                    valid_counts[serial] += 1
                    consecutive_missing[serial] = 0
                else:
                    missing_frames[serial].append(attempts)
                    consecutive_missing[serial] += 1
                    longest_missing[serial] = max(
                        longest_missing[serial], consecutive_missing[serial]
                    )
                    missing.append(serial)
            if missing:
                print(
                    f"  MISSING frame={attempts} t={elapsed:.3f}s "
                    f"trackers={', '.join(sorted(missing))}",
                    flush=True,
                )
            time.sleep(0.01)
        print(f"Pose health over {attempts} reads:")
        for serial in sorted(devices):
            rate = 100.0 * valid_counts[serial] / attempts if attempts else 0.0
            print(
                f"  {serial}: valid={valid_counts[serial]}/{attempts} "
                f"({rate:.2f}%) missing={len(missing_frames[serial])} "
                f"longest_missing_run={longest_missing[serial]} reads"
            )
            frames = ", ".join(str(frame) for frame in missing_frames[serial])
            print(f"    missing_frames: [{frames}]")
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
