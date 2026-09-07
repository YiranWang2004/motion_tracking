"""Batched ScaleBFM tracking of a stationary DefaultPose reference."""
from __future__ import annotations
import numpy as np
from omnicontact.contracts import PDCommand, RobotPose
from .scalebfm_policy import ScaleBFMHistory


class DualScaleBFMStanding:
    def __init__(self, task_policy, default_command):
        # Share model weights/FK, but never task history, residuals or frame state.
        self.policy = task_policy
        self.default_command = default_command
        self.histories = (ScaleBFMHistory(), ScaleBFMHistory())
        self.previous_action = np.zeros((2, 29), dtype=np.float32)
        self.reference_pos = None
        self.reference_quat = None
        self.reference_height = None

    def reset(self, states, snapshot):
        poses = (snapshot.robot_a, snapshot.robot_b)
        if self.reference_height is None:
            self.reference_height = [float(p.position_w[2]) for p in poses]
        targets = []
        for index, pose in enumerate(poses):
            x, y, z, w = pose.quaternion_xyzw
            yaw = np.arctan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
            position = pose.position_w.copy()
            # Re-anchor XY/yaw after a task, retaining the original standing
            # height in calibrated world coordinates rather than a crouched Z.
            position[2] = self.reference_height[index]
            target_pose = RobotPose(position, np.array([0.,0.,np.sin(yaw/2),np.cos(yaw/2)]), pose.stamp_s)
            targets.append(self.policy.kinematics[index].forward(self.default_command.target_pos, target_pose))
        self.reference_pos = np.repeat(np.stack([t.body_pos_w for t in targets])[:,None],6,axis=1)
        self.reference_quat = np.repeat(np.stack([t.body_quat_wxyz for t in targets])[:,None],6,axis=1)
        self.previous_action.fill(0)
        for history in self.histories:
            history.reset()

    def compute(self, states, snapshot):
        if self.reference_pos is None:
            raise RuntimeError('standing reference has not been initialized')
        poses = (snapshot.robot_a, snapshot.robot_b)
        live = [fk.forward(state.q_lab, pose) for fk,state,pose in zip(self.policy.kinematics,states,poses)]
        for index,(history,state) in enumerate(zip(self.histories,states)):
            history.update(state.quat_wxyz,state.gyro,state.q_lab,state.dq_lab,self.previous_action[index])
        target, action = self.policy.scalebfm.infer_batch(
            self.histories,self.reference_pos,self.reference_quat,
            np.stack([v.body_pos_w for v in live]),np.stack([v.body_quat_wxyz for v in live]),
            control_mode=self.policy.control_mode,time_offsets=self.policy.time_offsets)
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(action)):
            raise RuntimeError('ScaleBFM standing produced non-finite output')
        self.previous_action = action.copy()
        return [PDCommand(q,self.policy.kp,self.policy.kd) for q in target]
