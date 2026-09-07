"""Validated dual-G1 CFGen/Kimodo training bundle loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omnicontact.contracts import RobotPose
from omnicontact.reference.math_wxyz import quat_apply_batch, quat_mul_left_batch

from .constants import ANCHOR_BODY_NAME, KEY_BODY_NAMES, POLICY_JOINT_NAMES


def _xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(4)[[3, 0, 1, 2]]


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        (np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)), dtype=np.float32
    )


@dataclass(frozen=True)
class ReferenceFrame:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_wxyz: np.ndarray
    anchor_ang_vel_w: np.ndarray
    object_pos_w: np.ndarray
    object_quat_wxyz: np.ndarray


class DualReferenceBundle:
    """One 50 Hz reference with both robots in policy/body training order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=False) as data:
            self.fps = float(np.asarray(data["fps"]).item())
            if int(np.asarray(data["training_robot_count"]).item()) != 2:
                raise ValueError("reference bundle must contain two training robots")
            joint_order = tuple(str(value) for value in data["training_joint_order"])
            body_order = tuple(str(value) for value in data["training_body_order"])
            if joint_order != POLICY_JOINT_NAMES:
                raise ValueError(
                    "reference joint order does not match deployment policy order"
                )
            missing = [name for name in KEY_BODY_NAMES if name not in body_order]
            if missing:
                raise ValueError(f"reference is missing ScaleBFM bodies: {missing}")
            self.body_order = body_order
            self.key_body_indices = np.asarray(
                [body_order.index(name) for name in KEY_BODY_NAMES], dtype=np.int64
            )
            self.anchor_index = body_order.index(ANCHOR_BODY_NAME)
            self.joint_pos = np.stack(
                [
                    np.asarray(
                        data[f"training_robot_{index}_joint_pos"], dtype=np.float32
                    )
                    for index in range(2)
                ]
            )
            self.joint_vel = np.stack(
                [
                    np.asarray(
                        data[f"training_robot_{index}_joint_vel"], dtype=np.float32
                    )
                    for index in range(2)
                ]
            )
            self.body_pos_w = np.stack(
                [
                    np.asarray(
                        data[f"training_robot_{index}_body_pos_w"], dtype=np.float32
                    )
                    for index in range(2)
                ]
            )
            self.body_quat_wxyz = np.stack(
                [
                    np.asarray(
                        data[f"training_robot_{index}_body_quat_w"], dtype=np.float32
                    )
                    for index in range(2)
                ]
            )
            self.body_ang_vel_w = np.stack(
                [
                    np.asarray(
                        data[f"training_robot_{index}_body_ang_vel_w"], dtype=np.float32
                    )
                    for index in range(2)
                ]
            )
            self.object_pos_w = np.asarray(
                data["training_object_body_pos_w"], dtype=np.float32
            )
            self.object_quat_wxyz = np.asarray(
                data["training_object_body_quat_w"], dtype=np.float32
            )
            self.box_half_extents = np.asarray(
                data["training_box_half_extents"], dtype=np.float32
            ).reshape(3)
        self.frames = int(self.joint_pos.shape[1])
        self._validate_shapes()
        self._aligned = False

    def _validate_shapes(self) -> None:
        expected = {
            "joint_pos": (2, self.frames, 29),
            "joint_vel": (2, self.frames, 29),
            "body_pos_w": (2, self.frames, len(self.body_order), 3),
            "body_quat_wxyz": (2, self.frames, len(self.body_order), 4),
            "body_ang_vel_w": (2, self.frames, len(self.body_order), 3),
            "object_pos_w": (self.frames, 3),
            "object_quat_wxyz": (self.frames, 4),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"invalid reference field {name}: {value.shape}, expected {shape}"
                )
        if self.fps <= 0.0 or self.frames < 6:
            raise ValueError(
                "reference must contain at least six frames at positive FPS"
            )
        if not np.isclose(self.fps, 50.0, atol=1.0e-3):
            raise ValueError(f"deployment requires a 50 Hz reference, got {self.fps}")
        if np.any(self.box_half_extents <= 0.0):
            raise ValueError("reference box half extents must be positive")

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps

    def align_to_robot_a(self, pose: RobotPose, *, mode: str = "xyyaw") -> None:
        """Apply one global yaw/translation to every world-frame reference."""

        if self._aligned:
            raise RuntimeError("reference alignment can only be applied once")
        if mode in ("none", "motion_world"):
            self._aligned = True
            return
        if mode != "xyyaw":
            raise ValueError("reference_alignment must be 'motion_world', 'none' (legacy), or 'xyyaw'")
        pelvis_index = self.body_order.index("pelvis")
        reference_position = self.body_pos_w[0, 0, pelvis_index]
        reference_quat = self.body_quat_wxyz[0, 0, pelvis_index]
        actual_quat = _xyzw_to_wxyz(pose.quaternion_xyzw)
        yaw_delta = _yaw_from_wxyz(actual_quat) - _yaw_from_wxyz(reference_quat)
        rotation = _yaw_quaternion(yaw_delta)
        rotated_reference_position = quat_apply_batch(
            rotation, reference_position.reshape(1, 3)
        )[0]
        translation = (
            np.asarray(pose.position_w, dtype=np.float32) - rotated_reference_position
        )

        flat_body_pos = self.body_pos_w.reshape(-1, 3)
        self.body_pos_w = (
            quat_apply_batch(rotation, flat_body_pos).reshape(self.body_pos_w.shape)
            + translation
        ).astype(np.float32)
        flat_body_quat = self.body_quat_wxyz.reshape(-1, 4)
        self.body_quat_wxyz = quat_mul_left_batch(rotation, flat_body_quat).reshape(
            self.body_quat_wxyz.shape
        )
        flat_ang_vel = self.body_ang_vel_w.reshape(-1, 3)
        self.body_ang_vel_w = quat_apply_batch(rotation, flat_ang_vel).reshape(
            self.body_ang_vel_w.shape
        )
        self.object_pos_w = (
            quat_apply_batch(rotation, self.object_pos_w) + translation
        ).astype(np.float32)
        self.object_quat_wxyz = quat_mul_left_batch(rotation, self.object_quat_wxyz)
        self._aligned = True

    def frame(self, index: int) -> tuple[ReferenceFrame, ReferenceFrame]:
        frame = int(np.clip(index, 0, self.frames - 1))
        result = []
        for robot in range(2):
            result.append(
                ReferenceFrame(
                    joint_pos=self.joint_pos[robot, frame],
                    joint_vel=self.joint_vel[robot, frame],
                    body_pos_w=self.body_pos_w[robot, frame, self.key_body_indices],
                    body_quat_wxyz=self.body_quat_wxyz[
                        robot, frame, self.key_body_indices
                    ],
                    anchor_ang_vel_w=self.body_ang_vel_w[
                        robot, frame, self.anchor_index
                    ],
                    object_pos_w=self.object_pos_w[frame],
                    object_quat_wxyz=self.object_quat_wxyz[frame],
                )
            )
        return result[0], result[1]

    def future_key_bodies(
        self, frame: int, offsets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.minimum(
            int(frame) + np.asarray(offsets, dtype=np.int64), self.frames - 1
        )
        return (
            self.body_pos_w[:, indices][:, :, self.key_body_indices],
            self.body_quat_wxyz[:, indices][:, :, self.key_body_indices],
        )
