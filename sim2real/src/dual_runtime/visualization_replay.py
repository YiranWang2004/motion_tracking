"""Offline reader for dual ScaleBFM deployment rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from omnicontact.contracts import RobotPose

from .reference import DualReferenceBundle
from .visualization import encode_visualization


class DualScaleBFMReplay:
    """Load a rollout and reconstruct the live visualization protocol."""

    def __init__(
        self,
        path: str | Path,
        *,
        reference_bundle: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        with np.load(self.path, allow_pickle=False) as data:
            self.arrays = {name: np.asarray(data[name]) for name in data.files}
        metadata_path = self.path.with_name("metadata.json")
        self.metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.metadata = raw
        required = {
            "time_ns": (None,),
            "q": (None, 2, 29),
            "robot_position_w": (None, 2, 3),
            "robot_quat_xyzw": (None, 2, 4),
            "object_position_w": (None, 3),
            "object_quat_xyzw": (None, 4),
            "command_target": (None, 2, 29),
            "frame": (None,),
        }
        for name, expected in required.items():
            if name not in self.arrays:
                raise ValueError(f"rollout is missing {name}: {self.path}")
            shape = self.arrays[name].shape
            if len(shape) != len(expected) or any(
                want is not None and got != want
                for got, want in zip(shape, expected)
            ):
                raise ValueError(f"invalid rollout field {name}: {shape}")
        self.frame_count = int(self.arrays["q"].shape[0])
        if self.frame_count < 1 or any(
            value.shape[0] != self.frame_count for value in self.arrays.values()
        ):
            raise ValueError("rollout fields must have the same non-zero length")
        timestamps = np.asarray(self.arrays["time_ns"], dtype=np.int64)
        self.elapsed_s = (timestamps - timestamps[0]).astype(np.float64) * 1.0e-9
        if np.any(np.diff(self.elapsed_s) < 0.0):
            raise ValueError("rollout timestamps must be monotonic")
        reference_value = reference_bundle or self.metadata.get("reference")
        if reference_value is None:
            raise ValueError("reference bundle is required for rollout replay")
        self.reference = DualReferenceBundle(reference_value)
        alignment = str(self.metadata.get("reference_alignment", "xyyaw"))
        recorded_pose = self.metadata.get("reference_alignment_pose")
        active = np.flatnonzero(self.arrays["frame"] >= 0) if "frame" in self.arrays else []
        first = int(active[0]) if len(active) else 0
        first_pose = RobotPose(
            np.asarray(recorded_pose["position_w"]) if recorded_pose else self.arrays["robot_position_w"][first, 0],
            np.asarray(recorded_pose["quaternion_xyzw"]) if recorded_pose else self.arrays["robot_quat_xyzw"][first, 0],
            0.0,
        )
        self.reference.align_to_robot_a(first_pose, mode=alignment)

    @property
    def duration_s(self) -> float:
        return float(self.elapsed_s[-1]) if self.frame_count > 1 else 0.0

    def frame_index_at(self, elapsed_s: float) -> int:
        """Return the rollout frame active at a replay-relative timestamp."""

        return int(
            np.clip(
                np.searchsorted(self.elapsed_s, float(elapsed_s), side="right") - 1,
                0,
                self.frame_count - 1,
            )
        )

    def visualization_packet(self, index: int) -> dict[str, Any]:
        index = int(np.clip(index, 0, self.frame_count - 1))
        recorded_frame = int(self.arrays["frame"][index])
        reference_index = max(
            int(self.metadata.get("start_frame", 1)), recorded_frame
        )
        reference_a, reference_b = self.reference.frame(reference_index)
        robot_base = np.concatenate(
            (
                np.asarray(self.arrays["robot_position_w"][index], dtype=np.float32),
                np.asarray(self.arrays["robot_quat_xyzw"][index], dtype=np.float32)[
                    :, [3, 0, 1, 2]
                ],
            ),
            axis=1,
        )
        object_pose = np.concatenate(
            (
                np.asarray(self.arrays["object_position_w"][index], dtype=np.float32),
                np.asarray(self.arrays["object_quat_xyzw"][index], dtype=np.float32)[
                    [3, 0, 1, 2]
                ],
            )
        )
        actual_extents = (
            self.arrays["object_half_extents"][index]
            if "object_half_extents" in self.arrays
            else self.reference.box_half_extents
        )
        optional: dict[str, np.ndarray | None] = {
            "scalebfm_target": None,
            "residual": None,
        }
        for field in optional:
            if field in self.arrays:
                value = np.asarray(self.arrays[field][index], dtype=np.float32)
                if value.shape == (2, 29) and np.all(np.isfinite(value)):
                    optional[field] = value
        state = self.arrays.get("deployment_state")
        deployment_state = "replay" if state is None else str(state[index])
        return encode_visualization(
            actual_robot_base_wxyz=robot_base,
            actual_object_wxyz=object_pose,
            actual_box_half_extents=actual_extents,
            reference_robot_base_wxyz=np.stack(
                [
                    np.concatenate(
                        (reference_a.body_pos_w[0], reference_a.body_quat_wxyz[0])
                    ),
                    np.concatenate(
                        (reference_b.body_pos_w[0], reference_b.body_quat_wxyz[0])
                    ),
                ]
            ),
            reference_joint_pos=np.stack(
                (reference_a.joint_pos, reference_b.joint_pos)
            ),
            reference_object_wxyz=np.concatenate(
                (reference_a.object_pos_w, reference_a.object_quat_wxyz)
            ),
            reference_box_half_extents=self.reference.box_half_extents,
            target_joint_pos=self.arrays["command_target"][index],
            deployment_state=deployment_state,
            frame=recorded_frame,
            scalebfm_target=optional["scalebfm_target"],
            residual=optional["residual"],
        )
