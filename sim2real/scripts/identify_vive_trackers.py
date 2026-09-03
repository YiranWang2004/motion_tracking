#!/usr/bin/env python3
"""Continuously print raw XYZ positions for every OpenVR Generic Tracker."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnicontact.perception.openvr_tracker import (  # noqa: E402
    OpenVRTrackerReader,
    ViveSample,
)


def _xyz(sample: ViveSample) -> tuple[float, float, float]:
    return sample.line_x_m, sample.line_y_m, sample.line_z_m


def _format_row(
    serial: str,
    index: int,
    sample: ViveSample | None,
    origin: tuple[float, float, float] | None,
) -> str:
    if sample is None:
        return f"{serial:<18} {index:>5}  {'INVALID':<7} {'-':>9} {'-':>9} {'-':>9} {'-':>10}"
    x, y, z = _xyz(sample)
    if origin is None:
        moved = 0.0
    else:
        moved = ((x - origin[0]) ** 2 + (y - origin[1]) ** 2 + (z - origin[2]) ** 2) ** 0.5
    return (
        f"{serial:<18} {index:>5}  {'VALID':<7} "
        f"{x:>+9.4f} {y:>+9.4f} {z:>+9.4f} {moved:>10.4f}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously display raw SteamVR XYZ for all Generic Trackers; "
            "move one Tracker at a time to identify its serial number"
        )
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--refresh-devices-s",
        type=float,
        default=1.0,
        help="how often to re-enumerate connected Trackers (default: 1.0)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="optional duration; by default run until Ctrl-C",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="append snapshots instead of refreshing one terminal table",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.refresh_devices_s <= 0.0:
        parser.error("--refresh-devices-s must be positive")
    if args.seconds is not None and args.seconds <= 0.0:
        parser.error("--seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reader = OpenVRTrackerReader()
    origins: dict[str, tuple[float, float, float]] = {}
    started = time.monotonic()
    next_refresh = started
    devices: dict[str, int] = {}
    clear_terminal = sys.stdout.isatty() and not args.no_clear

    try:
        devices = reader.start()
        while True:
            loop_started = time.monotonic()
            if args.seconds is not None and loop_started - started >= args.seconds:
                break
            if loop_started >= next_refresh:
                devices = reader.refresh_devices()
                next_refresh = loop_started + args.refresh_devices_s

            serials = tuple(sorted(devices))
            samples = reader.read_all(serials)
            for serial, sample in samples.items():
                if sample is not None and serial not in origins:
                    origins[serial] = _xyz(sample)

            if clear_terminal:
                print("\033[H\033[J", end="")
            print(
                "OpenVR Generic Trackers — raw TrackingUniverseStanding coordinates\n"
                "Move only one Tracker at a time; the changing row identifies its serial.\n"
                f"detected={len(devices)}  elapsed={loop_started - started:.1f}s  Ctrl-C to stop\n"
            )
            print(
                f"{'SERIAL':<18} {'INDEX':>5}  {'POSE':<7} "
                f"{'X (m)':>9} {'Y (m)':>9} {'Z (m)':>9} {'MOVED (m)':>10}"
            )
            print("-" * 84)
            if not devices:
                print("No Generic Trackers detected. Check power, dongles, and pairing.")
            else:
                for serial in serials:
                    print(
                        _format_row(
                            serial,
                            devices[serial],
                            samples.get(serial),
                            origins.get(serial),
                        )
                    )
            sys.stdout.flush()

            remaining = 1.0 / args.fps - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
