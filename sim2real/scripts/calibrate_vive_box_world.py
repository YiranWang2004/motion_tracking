#!/usr/bin/env python3
"""Calibrate the task world from one box Tracker placement.

This is a deliberately simple alternative to the O/+X/+Y calibration.  The
box Tracker is placed at the center of the top face, with its local Z axis
pointing down and its X/Y axes parallel to the box edges.  The box center in
the task world is supplied by the operator; the script then solves
``world_from_steamvr`` from one averaged OpenVR pose.

The initial Tracker-to-box transform is intentionally a rough placeholder:
the Tracker origin is assumed to be ``box_top_z - box_center_z`` above the
box center, and the transform is a 180-degree rotation around X (xyzw
quaternion ``[1, 0, 0, 0]``).  Refine this value in the deployment JSON after
mechanical measurements are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_vive_world import capture_pose
from omnicontact.perception.openvr_tracker import OpenVRTrackerReader
from omnicontact.perception.vive_pose import RigidTransform


def _quat_inverse(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    return np.array(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        dtype=np.float64,
    )


def _transform_inverse(transform: RigidTransform) -> RigidTransform:
    """Return the inverse of a ``parent_from_child`` rigid transform."""
    inverse_quaternion = _quat_inverse(transform.quaternion_xyzw)
    # Rotating the negative parent-frame translation by the inverse rotation
    # gives the child-frame origin expressed in the inverse parent frame.
    inverse_rotation = RigidTransform(
        np.zeros(3, dtype=np.float64), inverse_quaternion
    )
    inverse_position = inverse_rotation.compose(
        RigidTransform(-transform.position, [0.0, 0.0, 0.0, 1.0])
    ).position
    return RigidTransform(inverse_position, inverse_quaternion)


def _transform_dict(transform: RigidTransform) -> dict[str, list[float]]:
    return {
        "position_m": [float(value) for value in transform.position],
        "quaternion_xyzw": [float(value) for value in transform.quaternion_xyzw],
    }


def solve_box_tracker_world(
    tracker_sample_transform: RigidTransform,
    box_center_world: np.ndarray,
    box_top_z: float,
) -> tuple[RigidTransform, RigidTransform]:
    """Solve ``world_from_steamvr`` and return the placeholder box extrinsic.

    Frames:
      ``tracker_sample_transform`` is ``^S T_T`` from OpenVR.
      The returned first transform is ``^W T_S``.
      The returned second transform is ``^T T_O`` (tracker parent, box child).
    """
    box_center = np.asarray(box_center_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(box_center)):
        raise ValueError("box_center_world must be finite")
    if not np.isfinite(box_top_z):
        raise ValueError("box_top_z must be finite")
    offset_z = float(box_top_z - box_center[2])
    if offset_z <= 0.0:
        raise ValueError("box_top_z must be above box_center_world.z")

    # ^T T_O: object center is below the Tracker by offset_z in Tracker +Z.
    # The 180-degree X rotation keeps X aligned while making Tracker +Z point
    # down relative to the box/world +Z; it also keeps the frame right-handed.
    tracker_from_object = RigidTransform(
        [0.0, 0.0, offset_z],
        [1.0, 0.0, 0.0, 0.0],
    )
    world_from_object = RigidTransform(
        box_center,
        [0.0, 0.0, 0.0, 1.0],
    )
    world_from_tracker = world_from_object.compose(
        _transform_inverse(tracker_from_object)
    )
    world_from_steamvr = world_from_tracker.compose(
        _transform_inverse(tracker_sample_transform)
    )
    return world_from_steamvr, tracker_from_object


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Vive config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Vive config must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate world_from_steamvr from a Tracker placed on the box"
    )
    parser.add_argument("--vive-config", required=True, help="input deployment JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="optional output JSON; without it only print the computed transforms",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="box Tracker serial; defaults to object_tracker_serial in the JSON",
    )
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument(
        "--box-center-world",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.15),
        metavar=("X", "Y", "Z"),
        help="known box center in the task world; default is [1, 0, 0.15] m",
    )
    parser.add_argument(
        "--box-top-z",
        type=float,
        default=0.30,
        help="world Z height of the box top / Tracker origin, default 0.30 m",
    )
    args = parser.parse_args(argv)
    if args.samples < 20:
        parser.error("--samples must be at least 20")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.vive_config).expanduser().resolve()
    config = _load_json(config_path)
    serial = args.serial or str(config.get("object_tracker_serial", "")).strip()
    if not serial:
        raise SystemExit("--serial is required when object_tracker_serial is absent")

    reader = OpenVRTrackerReader([serial])
    try:
        devices = reader.start()
        if serial not in devices:
            raise RuntimeError(
                f"box Tracker {serial!r} was not found; detected={sorted(devices)}"
            )
        print(f"Sampling box Tracker {serial!r}; keep it still and do not rotate it")
        capture = capture_pose(reader, serial, args.samples)
        tracker_sample_transform = RigidTransform(
            capture.position_s,
            capture.quaternion_s_tracker_xyzw,
        )
    finally:
        reader.stop()

    world_from_steamvr, tracker_from_object = solve_box_tracker_world(
        tracker_sample_transform,
        np.asarray(args.box_center_world, dtype=np.float64),
        float(args.box_top_z),
    )
    print("\n=== box Tracker calibration ===")
    print("world_from_steamvr (^W T_S):")
    print(json.dumps(_transform_dict(world_from_steamvr), indent=2))
    print("object_tracker_to_object placeholder (^T T_O):")
    print(json.dumps(_transform_dict(tracker_from_object), indent=2))
    print(
        "\nAssumption: Tracker +X is box +X, Tracker +Z points down, "
        "and box center is directly below the Tracker."
    )

    if args.output is None:
        return 0

    output_path = Path(args.output).expanduser().resolve()
    config["world_from_steamvr"] = _transform_dict(world_from_steamvr)
    config["object_tracker_to_object"] = _transform_dict(tracker_from_object)
    _write_json(output_path, config)
    print(f"Wrote updated Vive config to {output_path}")
    if config.get("calibration_confirmed") is not True:
        print(
            "WARNING: calibration_confirmed is not true; inspect the transforms "
            "and set it to true before deployment."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
