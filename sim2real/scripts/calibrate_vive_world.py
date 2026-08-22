#!/usr/bin/env python3
"""Solve a SteamVR-standing to user-world transform from three placements."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnicontact.perception.openvr_tracker import OpenVRTrackerReader, ViveSample
from omnicontact.perception.vive_pose import RigidTransform, sample_to_transform


@dataclass(frozen=True)
class Capture:
    position_s: np.ndarray
    quaternion_s_tracker_xyzw: np.ndarray
    position_std_mm: np.ndarray
    orientation_std_deg: float
    valid_samples: int


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def solve_world_from_steamvr(
    origin_s: np.ndarray,
    positive_x_s: np.ndarray,
    positive_y_s: np.ndarray,
    origin_w: np.ndarray,
) -> tuple[RigidTransform, dict[str, float]]:
    """Fit W_from_S while mapping the two measured motions to W +X/+Y."""
    origin_s = np.asarray(origin_s, dtype=np.float64).reshape(3)
    origin_w = np.asarray(origin_w, dtype=np.float64).reshape(3)
    delta_x = np.asarray(positive_x_s, dtype=np.float64).reshape(3) - origin_s
    delta_y = np.asarray(positive_y_s, dtype=np.float64).reshape(3) - origin_s
    distance_x = float(np.linalg.norm(delta_x))
    distance_y = float(np.linalg.norm(delta_y))
    if distance_x < 0.05 or distance_y < 0.05:
        raise ValueError("calibration motions must each be at least 5 cm")

    x_axis_in_s = delta_x / distance_x
    raw_cosine = float(np.dot(delta_x, delta_y) / (distance_x * distance_y))
    y_orthogonal = delta_y - np.dot(delta_y, x_axis_in_s) * x_axis_in_s
    y_norm = float(np.linalg.norm(y_orthogonal))
    if y_norm < 0.05:
        raise ValueError("+X and +Y calibration motions are nearly parallel")
    y_axis_in_s = y_orthogonal / y_norm
    z_axis_in_s = np.cross(x_axis_in_s, y_axis_in_s)

    # Columns are the desired W basis vectors expressed in SteamVR S.
    rotation_s_from_w = np.column_stack(
        (x_axis_in_s, y_axis_in_s, z_axis_in_s)
    )
    rotation_w_from_s = rotation_s_from_w.T
    position_w_from_s = origin_w - rotation_w_from_s @ origin_s
    transform = RigidTransform(
        position_w_from_s,
        _matrix_to_quaternion_xyzw(rotation_w_from_s),
    )
    diagnostics = {
        "measured_x_distance_m": distance_x,
        "measured_y_distance_m": distance_y,
        "raw_xy_angle_deg": math.degrees(math.acos(float(np.clip(raw_cosine, -1.0, 1.0)))),
    }
    return transform, diagnostics


def capture_pose(
    reader: OpenVRTrackerReader,
    serial: str,
    sample_count: int,
) -> Capture:
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    deadline = time.monotonic() + max(5.0, sample_count * 0.05)
    while len(positions) < sample_count and time.monotonic() < deadline:
        sample: ViveSample | None = reader.read_all([serial]).get(serial)
        if sample is not None:
            positions.append([sample.line_x_m, sample.line_y_m, sample.line_z_m])
            quaternions.append([sample.qx, sample.qy, sample.qz, sample.qw])
        time.sleep(0.01)
    if len(positions) < sample_count:
        raise RuntimeError(
            f"only {len(positions)}/{sample_count} valid samples; check tracking"
        )

    position_array = np.asarray(positions, dtype=np.float64)
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    reference = quaternion_array[0]
    quaternion_array[np.sum(quaternion_array * reference, axis=1) < 0.0] *= -1.0
    mean_quaternion = quaternion_array.mean(axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    dots = np.clip(np.abs(quaternion_array @ mean_quaternion), 0.0, 1.0)
    orientation_errors = np.degrees(2.0 * np.arccos(dots))
    return Capture(
        position_s=position_array.mean(axis=0),
        quaternion_s_tracker_xyzw=mean_quaternion,
        position_std_mm=position_array.std(axis=0) * 1000.0,
        orientation_std_deg=float(np.std(orientation_errors)),
        valid_samples=len(position_array),
    )


def _capture_at_prompt(
    reader: OpenVRTrackerReader,
    serial: str,
    sample_count: int,
    prompt: str,
) -> Capture:
    input(f"\n{prompt}\n保持不动后按 Enter 开始采样... ")
    capture = capture_pose(reader, serial, sample_count)
    print(
        f"  S position = {capture.position_s.tolist()} m\n"
        f"  position std = {capture.position_std_mm.tolist()} mm\n"
        f"  orientation jitter std = {capture.orientation_std_deg:.4f} deg"
    )
    return capture


def _transform_dict(transform: RigidTransform) -> dict[str, list[float]]:
    return {
        "position_m": transform.position.tolist(),
        "quaternion_xyzw": transform.quaternion_xyzw.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate SteamVR Standing frame from origin/+X/+Y placements"
    )
    parser.add_argument("--serial", default=None)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--distance-x", type=float, default=0.50)
    parser.add_argument("--distance-y", type=float, default=0.50)
    parser.add_argument(
        "--origin-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="world coordinates assigned to the Tracker device origin at placement O",
    )
    args = parser.parse_args()
    if args.samples < 20:
        raise SystemExit("--samples must be at least 20")
    if args.distance_x <= 0.05 or args.distance_y <= 0.05:
        raise SystemExit("calibration distances must exceed 5 cm")

    reader = OpenVRTrackerReader([args.serial] if args.serial else None)
    try:
        devices = reader.start()
        if args.serial is not None:
            serial = args.serial
        elif len(devices) == 1:
            serial = next(iter(devices))
        else:
            choices = ", ".join(sorted(devices))
            raise RuntimeError(
                f"found multiple Trackers ({choices}); specify --serial"
            )

        print(f"Tracker: {serial}")
        print(
            "整个实验中不要旋转 Tracker。+X/+Y 是你希望定义的世界方向，"
            "例如机器人前方/左方。"
        )
        origin = _capture_at_prompt(
            reader,
            serial,
            args.samples,
            "位置 O：把 Tracker 放在世界原点标记处；这也是 Tracker 启动帧 T0。",
        )
        positive_x = _capture_at_prompt(
            reader,
            serial,
            args.samples,
            f"位置 X：从 O 平移到世界 +X 约 {args.distance_x:.3f} m。",
        )
        positive_y = _capture_at_prompt(
            reader,
            serial,
            args.samples,
            f"位置 Y：回到 O，再平移到世界 +Y 约 {args.distance_y:.3f} m。",
        )

        world_from_steamvr, diagnostics = solve_world_from_steamvr(
            origin.position_s,
            positive_x.position_s,
            positive_y.position_s,
            np.asarray(args.origin_world),
        )
        steamvr_from_tracker_start = RigidTransform(
            origin.position_s,
            origin.quaternion_s_tracker_xyzw,
        )
        world_from_tracker_start = world_from_steamvr.compose(
            steamvr_from_tracker_start
        )
        result = {
            "tracker_serial": serial,
            "world_from_steamvr": _transform_dict(world_from_steamvr),
            "steamvr_from_tracker_start": _transform_dict(
                steamvr_from_tracker_start
            ),
            "world_from_tracker_start": _transform_dict(
                world_from_tracker_start
            ),
            "diagnostics": diagnostics,
        }
        print("\n=== calibration result ===")
        print(json.dumps(result, indent=2))
        print(
            "\n检查 measured distances 是否接近尺量值、raw_xy_angle_deg 是否接近 90°。"
        )
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
