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
        missing_intervals = {serial: [] for serial in devices}
        missing_since = {serial: None for serial in devices}
        attempts = 0
        start = time.monotonic()
        end = start + args.seconds
        print(f"Checking poses for {args.seconds:g}s...", flush=True)
        try:
            while time.monotonic() < end:
                samples = reader.read_all()
                attempts += 1
                elapsed = time.monotonic() - start
                for serial in devices:
                    if samples.get(serial) is not None:
                        valid_counts[serial] += 1
                        if missing_since[serial] is not None:
                            missing_intervals[serial].append(
                                (missing_since[serial], elapsed)
                            )
                            missing_since[serial] = None
                    elif missing_since[serial] is None:
                        missing_since[serial] = elapsed
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nInterrupted; summarizing collected reads.")
        elapsed = time.monotonic() - start
        print(f"Pose health over {attempts} reads:")
        for serial in sorted(devices):
            rate = 100.0 * valid_counts[serial] / attempts if attempts else 0.0
            print(
                f"  {serial}: valid={valid_counts[serial]}/{attempts} "
                f"({rate:.2f}%) missing={attempts - valid_counts[serial]}"
            )
            for interval_start, interval_end in missing_intervals[serial]:
                print(
                    f"    MISSING {interval_start:.3f}s - {interval_end:.3f}s "
                    f"duration={interval_end - interval_start:.3f}s"
                )
            if missing_since[serial] is not None:
                print(
                    f"    MISSING {missing_since[serial]:.3f}s - {elapsed:.3f}s "
                    f"duration={elapsed - missing_since[serial]:.3f}s "
                    "(still missing at end)"
                )
            elif not missing_intervals[serial]:
                print("    No missing intervals.")
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
