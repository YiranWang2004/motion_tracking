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
        attempts = 0
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            samples = reader.read_all()
            attempts += 1
            for serial, sample in samples.items():
                valid_counts[serial] += int(sample is not None)
            time.sleep(0.01)
        print(f"Pose health over {attempts} reads:")
        for serial in sorted(devices):
            print(f"  {serial}: valid={valid_counts[serial]}/{attempts}")
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
