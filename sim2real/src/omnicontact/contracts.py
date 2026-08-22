"""Stable task data contracts shared by simulation, Vive, and policies."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectPose:
    """A rigid-object pose expressed in RoboJuDo's world/odometry frame."""

    position_w: np.ndarray
    quaternion_xyzw: np.ndarray
    half_extents: np.ndarray
    stamp_s: float
    confidence: float = 1.0
    linear_velocity_w: np.ndarray | None = None
    angular_velocity_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position_w, dtype=np.float32).reshape(3)
        quat = np.asarray(self.quaternion_xyzw, dtype=np.float32).reshape(4)
        extents = np.asarray(self.half_extents, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("Object position must be finite.")
        if not np.all(np.isfinite(quat)):
            raise ValueError("Object quaternion must be finite.")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            raise ValueError("Object quaternion must be non-zero.")
        if not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
            raise ValueError("Object half_extents must be positive.")
        if not math.isfinite(self.stamp_s):
            raise ValueError("Object timestamp must be finite.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Object confidence must be in [0, 1].")
        object.__setattr__(self, "position_w", position.copy())
        object.__setattr__(self, "quaternion_xyzw", (quat / norm).copy())
        object.__setattr__(self, "half_extents", extents.copy())
        for field in ("linear_velocity_w", "angular_velocity_w"):
            value = getattr(self, field)
            if value is not None:
                velocity = np.asarray(value, dtype=np.float32).reshape(3)
                if not np.all(np.isfinite(velocity)):
                    raise ValueError(f"Object {field} must be finite.")
                object.__setattr__(self, field, velocity.copy())


@dataclass(frozen=True)
class RobotPose:
    """Robot pelvis pose in the same calibrated world frame as ObjectPose."""

    position_w: np.ndarray
    quaternion_xyzw: np.ndarray
    stamp_s: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_w, dtype=np.float32).reshape(3)
        quat = np.asarray(self.quaternion_xyzw, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(position)):
            raise ValueError("Robot position must be finite.")
        if not np.all(np.isfinite(quat)):
            raise ValueError("Robot quaternion must be finite.")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            raise ValueError("Robot quaternion must be non-zero.")
        if not math.isfinite(self.stamp_s):
            raise ValueError("Robot timestamp must be finite.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Robot confidence must be in [0, 1].")
        object.__setattr__(self, "position_w", position.copy())
        object.__setattr__(self, "quaternion_xyzw", (quat / norm).copy())


@dataclass(frozen=True)
class TaskGoal:
    """Desired object center pose in the same world frame as ObjectPose."""

    position_w: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_w, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("Task goal position must be finite.")
        object.__setattr__(self, "position_w", position.copy())


@dataclass(frozen=True)
class PDCommand:
    """Absolute joint target and per-joint PD gains in environment joint order."""

    target_pos: np.ndarray
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    hand_pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        target = np.asarray(self.target_pos, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(target)):
            raise ValueError("target_pos must be finite.")
        object.__setattr__(self, "target_pos", target)
        for field in ("kp", "kd"):
            value = getattr(self, field)
            if value is not None:
                gain = np.asarray(value, dtype=np.float32).reshape(-1)
                if (
                    gain.shape != target.shape
                    or not np.all(np.isfinite(gain))
                    or np.any(gain < 0.0)
                ):
                    raise ValueError(f"{field} must be non-negative and match target_pos.")
                object.__setattr__(self, field, gain)


@dataclass(frozen=True)
class ReferenceVisualization:
    """World-frame reference snapshot consumed by simulation visualization."""

    left_wrist_wxyz: np.ndarray
    right_wrist_wxyz: np.ndarray
    torso_wxyz: np.ndarray
    left_ankle_wxyz: np.ndarray
    right_ankle_wxyz: np.ndarray
    object_wxyz: np.ndarray
    contact: np.ndarray
    ghost_base_wxyz: np.ndarray | None = None
    ghost_dof_pos: np.ndarray | None = None

    def __post_init__(self) -> None:
        for field in (
            "left_wrist_wxyz",
            "right_wrist_wxyz",
            "torso_wxyz",
            "left_ankle_wxyz",
            "right_ankle_wxyz",
            "object_wxyz",
        ):
            value = np.asarray(getattr(self, field), dtype=np.float32).reshape(7).copy()
            object.__setattr__(self, field, value)
        contact = np.asarray(self.contact, dtype=np.float32).reshape(4).copy()
        object.__setattr__(self, "contact", contact)
        if self.ghost_base_wxyz is not None:
            object.__setattr__(
                self,
                "ghost_base_wxyz",
                np.asarray(self.ghost_base_wxyz, dtype=np.float32).reshape(7).copy(),
            )
        if self.ghost_dof_pos is not None:
            object.__setattr__(
                self,
                "ghost_dof_pos",
                np.asarray(self.ghost_dof_pos, dtype=np.float32).reshape(-1).copy(),
            )

