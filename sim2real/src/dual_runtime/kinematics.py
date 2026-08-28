"""Per-robot MuJoCo FK for live ScaleBFM body and residual anchor features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omnicontact.contracts import RobotPose
from omnicontact.reference.mujoco_kinematics_wxyz import MujocoKinematics

from .constants import ANCHOR_BODY_NAME, KEY_BODY_NAMES, POLICY_JOINT_NAMES


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion, dtype=np.float32).reshape(4)[[3, 0, 1, 2]]


@dataclass(frozen=True)
class LiveKinematics:
    body_pos_w: np.ndarray
    body_quat_wxyz: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_wxyz: np.ndarray


class G1PolicyKinematics:
    def __init__(self, xml_path: str | Path) -> None:
        path = Path(xml_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.kinematics = MujocoKinematics(
            path.as_posix(), joint_names=list(POLICY_JOINT_NAMES)
        )
        missing = [
            name for name in KEY_BODY_NAMES if name not in self.kinematics.body_names
        ]
        if missing:
            raise ValueError(f"kinematics XML is missing ScaleBFM bodies: {missing}")

    def forward(self, joint_pos: np.ndarray, robot_pose: RobotPose) -> LiveKinematics:
        state = self.kinematics.forward(
            np.asarray(joint_pos, dtype=np.float32).reshape(29),
            np.asarray(robot_pose.position_w, dtype=np.float32),
            xyzw_to_wxyz(robot_pose.quaternion_xyzw),
        )
        body_pos = np.stack([state[name]["pos"] for name in KEY_BODY_NAMES]).astype(
            np.float32
        )
        body_quat = np.stack([state[name]["quat"] for name in KEY_BODY_NAMES]).astype(
            np.float32
        )
        anchor = state[ANCHOR_BODY_NAME]
        return LiveKinematics(
            body_pos_w=body_pos,
            body_quat_wxyz=body_quat,
            anchor_pos_w=np.asarray(anchor["pos"], dtype=np.float32),
            anchor_quat_wxyz=np.asarray(anchor["quat"], dtype=np.float32),
        )
