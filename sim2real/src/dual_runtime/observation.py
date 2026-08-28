"""Exact 201-D decentralized residual observation construction."""

from __future__ import annotations

import numpy as np

from omnicontact.contracts import ObjectPose
from omnicontact.reference.math_wxyz import (
    matrix_from_quat,
    quat_rotate_inverse,
    subtract_frame_transforms,
)
from omnicontact.runtime import BridgeState

from .kinematics import LiveKinematics, xyzw_to_wxyz
from .reference import ReferenceFrame


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(
        np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
    ).astype(np.float32)


def relative_pose(
    origin_pos: np.ndarray,
    origin_quat_wxyz: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position, quaternion = subtract_frame_transforms(
        origin_pos, origin_quat_wxyz, target_pos, target_quat_wxyz
    )
    orientation = matrix_from_quat(quaternion)[:, :2].reshape(-1)
    return position.astype(np.float32), orientation.astype(np.float32)


def build_residual_observation(
    *,
    state: BridgeState,
    reference: ReferenceFrame,
    live: LiveKinematics,
    partner_live: LiveKinematics,
    object_pose: ObjectPose,
    scalebfm_target: np.ndarray,
    previous_residual: np.ndarray,
    default_q: np.ndarray,
) -> np.ndarray:
    partner_position, partner_orientation = relative_pose(
        live.anchor_pos_w,
        live.anchor_quat_wxyz,
        partner_live.anchor_pos_w,
        partner_live.anchor_quat_wxyz,
    )
    object_position, object_orientation = relative_pose(
        live.anchor_pos_w,
        live.anchor_quat_wxyz,
        object_pose.position_w,
        xyzw_to_wxyz(object_pose.quaternion_xyzw),
    )
    observation = np.concatenate(
        (
            reference.joint_pos,
            reference.joint_vel,
            projected_gravity(state.quat_wxyz),
            reference.anchor_ang_vel_w,
            state.gyro,
            state.q_lab - np.asarray(default_q, dtype=np.float32),
            state.dq_lab,
            np.asarray(previous_residual, dtype=np.float32).reshape(29),
            partner_position,
            partner_orientation,
            np.asarray(scalebfm_target, dtype=np.float32).reshape(29),
            object_position,
            object_orientation,
        )
    ).astype(np.float32)
    if observation.shape != (201,) or not np.all(np.isfinite(observation)):
        raise RuntimeError(f"invalid residual observation: {observation.shape}")
    return observation
