#!/usr/bin/env python3
"""Set a fixed box-axis mapping without changing the calibrated world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omnicontact.perception.vive_pose import RigidTransform


def configure_box_axes(config: dict, quarter_turns_z: int) -> dict:
    """Compose a fixed local rotation with the calibrated mounting transform.

    Store the base separately so repeated calls replace, rather than accumulate,
    the axis mapping. Translation is expressed in the tracker and is unchanged.
    """
    definition = config.get("object_frame_definition", {})
    base = definition.get("tracker_from_axis_aligned_box", config["object_tracker_to_object"])
    transform = RigidTransform.from_dict(base, "tracker_from_axis_aligned_box")
    calibration = config.get("box_world_calibration")
    base_source = definition.get("base_source", "existing_mounting_transform")
    if calibration is not None:
        if calibration["tracker_serial"] != config["object_tracker_serial"]:
            raise ValueError("box calibration Tracker serial does not match object Tracker")
        # At calibration the borrowed Tracker's nominal world orientation is
        # Rx(pi). Only a small deviation may be treated as mounting correction;
        # a quarter-turn belongs in the explicit axis mapping below.
        anchor = RigidTransform.from_dict(
            calibration["world_from_tracker_start"], "world_from_tracker_start")
        q = anchor.quaternion_xyzw
        correction_deg = np.degrees(2 * np.arccos(np.clip(abs(q[0]), 0, 1)))
        if correction_deg > 10:
            raise ValueError(f"calibration mounting correction {correction_deg:.2f} deg exceeds 10 deg; check world axes")
        transform = RigidTransform(transform.position, [-q[0], -q[1], -q[2], q[3]])
        base = {"position_m": transform.position.tolist(),
                "quaternion_xyzw": transform.quaternion_xyzw.tolist()}
        base_source = "box_world_calibration"
    angle = (quarter_turns_z % 4) * np.pi / 2
    axes = RigidTransform([0, 0, 0], [0, 0, np.sin(angle / 2), np.cos(angle / 2)])
    effective = transform.compose(axes)
    config["object_frame_definition"] = {
        "tracker_from_axis_aligned_box": base,
        "box_local_z_quarter_turns": quarter_turns_z,
        "base_source": base_source,
    }
    config["object_tracker_to_object"] = {
        "position_m": effective.position.tolist(),
        "quaternion_xyzw": effective.quaternion_xyzw.tolist(),
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vive-config", required=True)
    parser.add_argument("--quarter-turns-z", required=True, type=int,
                        help="fixed local Z rotation in 90-degree units; replaces previous mapping")
    parser.add_argument("--output", help="write a derived config; otherwise only print")
    args = parser.parse_args()
    config = json.loads(Path(args.vive_config).expanduser().read_text())
    configure_box_axes(config, args.quarter_turns_z)
    result = json.dumps(config, indent=2) + "\n"
    if args.output:
        Path(args.output).expanduser().write_text(result)
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
