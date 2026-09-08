"""Single-side inference with tracker-only inter-robot observations."""
from dataclasses import dataclass
import time

import numpy as np

from omnicontact.contracts import PDCommand, RobotPose
from .observation import build_pelvis_residual_observation
from .scalebfm_policy import ScaleBFMHistory
from .scalebfm_residual_policy import DualScaleBFMResidualPolicy


@dataclass(frozen=True)
class OnboardStep:
    target: np.ndarray
    scalebfm_target: np.ndarray
    residual: np.ndarray
    observation: np.ndarray
    frame: int
    complete: bool
    inference_time_s: float


class OnboardPolicy(DualScaleBFMResidualPolicy):
    """Reuse artifact loading/reference preflight; execute only one robot row.

    No partner BridgeState is constructed or read. Both reference trajectories
    remain local for geometry preflight. The parent owns the shared model and
    checkpoint loader; only this side's FK, history and Actor execute.
    """
    def __init__(self, *, robot_id, anchor_angular_velocity_frame="reference-anchor", **kwargs):
        if robot_id not in {"a", "b"}:
            raise ValueError("robot_id must be a or b")
        if anchor_angular_velocity_frame not in {"reference-anchor", "world"}:
            raise ValueError("invalid reference angular velocity frame")
        super().__init__(interaction_frame="pelvis",
                         anchor_angular_velocity_frame=anchor_angular_velocity_frame, **kwargs)
        if self.reference_alignment != "motion_world":
            raise ValueError("onboard deployment requires shared motion_world alignment")
        self.index = 0 if robot_id == "a" else 1
        self.anchor_angular_velocity_frame = anchor_angular_velocity_frame

    def warmup(self):
        """Exercise FK and both networks before opening a bridge."""
        from omnicontact.contracts import ObjectPose
        from omnicontact.runtime import BridgeState
        from .dual_pose_provider import DualPoseSnapshot
        refs = self.reference.frame(self.start_frame)
        poses = [RobotPose(r.body_pos_w[0], r.body_quat_wxyz[0][[1, 2, 3, 0]],
                           time.monotonic()) for r in refs]
        snapshot = DualPoseSnapshot(*poses, ObjectPose(
            refs[0].object_pos_w, refs[0].object_quat_wxyz[[1, 2, 3, 0]],
            self.reference.box_half_extents, time.monotonic()))
        r = refs[self.index]
        state = BridgeState(r.joint_pos, r.joint_vel, r.body_quat_wxyz[0],
                            np.zeros(3), {}, None, 0, time.monotonic_ns())
        self.initialized = True
        try:
            for _ in range(5):
                self.compute_single(state, snapshot)
        finally:
            self.initialized = False
            self.reset_rollout()

    def compute_single(self, state, snapshot):
        if not self.initialized:
            raise RuntimeError("initialize before task inference")
        started = time.perf_counter()
        i = self.index
        poses = (snapshot.robot_a, snapshot.robot_b)
        live = self.kinematics[i].forward(state.q_lab, poses[i])
        history = self.histories[i]
        history.update(state.quat_wxyz, state.gyro, state.q_lab, state.dq_lab,
                       self.previous_executed_action[i])
        positions, quaternions = self.reference.future_key_bodies(self.frame, self.time_offsets)
        target, action = self.scalebfm.infer_batch(
            (history,), positions[i:i+1], quaternions[i:i+1],
            live.body_pos_w[None], live.body_quat_wxyz[None],
            control_mode=self.control_mode, time_offsets=self.time_offsets,
        )
        observation = build_pelvis_residual_observation(
            state=state, reference=self.reference.frame(self.frame)[i],
            own_pelvis=poses[i], partner_pelvis=poses[1-i], object_pose=snapshot.object,
            scalebfm_target=target[0], previous_residual=self.previous_residual[i],
            default_q=self.default_q,
            anchor_angular_velocity_frame=self.anchor_angular_velocity_frame,
        )
        residual = self.residual.infer_agent(observation, i)
        combined = target[0] + self.residual_scale * residual
        if not np.isfinite(combined).all():
            raise RuntimeError("non-finite onboard target")
        self.previous_residual[i] = residual
        self.previous_executed_action[i] = action[0] + self.residual_scale * residual / self.scalebfm.action_scale
        frame = self.frame
        complete = frame == self.reference.frames - 1
        if not complete:
            self.frame += 1
        return OnboardStep(combined, target[0], residual, observation, frame,
                           complete, time.perf_counter() - started)


class OnboardStanding:
    def __init__(self, policy, default_command):
        self.policy = policy
        self.default = default_command
        self.history = ScaleBFMHistory()
        self.action = np.zeros(29, dtype=np.float32)
        self.height = None

    def reset(self, snapshot):
        pose = (snapshot.robot_a, snapshot.robot_b)[self.policy.index]
        if self.height is None:
            self.height = float(pose.position_w[2])
        x, y, z, w = pose.quaternion_xyzw
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        position = pose.position_w.copy()
        position[2] = self.height
        reference = RobotPose(position, [0, 0, np.sin(yaw/2), np.cos(yaw/2)], pose.stamp_s)
        live = self.policy.kinematics[self.policy.index].forward(self.default.target_pos, reference)
        self.positions = np.repeat(live.body_pos_w[None, None], 6, axis=1)
        self.quaternions = np.repeat(live.body_quat_wxyz[None, None], 6, axis=1)
        self.history.reset()
        self.action.fill(0)

    def compute(self, state, snapshot):
        p = self.policy
        pose = (snapshot.robot_a, snapshot.robot_b)[p.index]
        live = p.kinematics[p.index].forward(state.q_lab, pose)
        self.history.update(state.quat_wxyz, state.gyro, state.q_lab, state.dq_lab, self.action)
        target, action = p.scalebfm.infer_batch(
            (self.history,), self.positions, self.quaternions,
            live.body_pos_w[None], live.body_quat_wxyz[None],
            control_mode=p.control_mode, time_offsets=p.time_offsets,
        )
        self.action = action[0].copy()
        return PDCommand(target[0], p.kp, p.kd)
