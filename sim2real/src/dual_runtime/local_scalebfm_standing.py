"""Tracker-free, pelvis-relative ScaleBFM reference hold and return to DefaultPose.

There is deliberately no world translation or heading error. FK uses a common
zero translation and IMU orientation; reference heading follows current IMU yaw.
This is local posture control, not global localization or foot odometry.
"""
import numpy as np

from omnicontact.contracts import PDCommand, RobotPose
from .scalebfm_policy import ScaleBFMHistory
from .vive_recovery import smoothstep


class LocalScaleBFMStanding:
    def __init__(self, policy, default_command, settings):
        self.policy, self.default, self.cfg = policy, default_command, settings
        self.history = ScaleBFMHistory()
        self.action = np.zeros(29, dtype=np.float32)
        self.reference_q = None
        self.return_from = None
        self.return_at = None
        self.saved_default = None

    def arm(self):
        if self.saved_default is None:
            self.saved_default = self.default.target_pos.copy()

    def hold(self, reference_q, previous_target):
        if self.saved_default is None:
            raise RuntimeError('local_return_requires_B_default_pose')
        self.reference_q = np.asarray(reference_q, dtype=np.float32).copy()
        self.return_at = None
        self.history.reset()
        self.executed(previous_target)

    def fallback(self, at):
        if self.return_at is None:
            self.return_from = self.reference_q.copy()
            self.return_at = at

    def executed(self, target):
        self.action = ((np.asarray(target)-self.policy.default_q)
                       / self.policy.scalebfm.action_scale).astype(np.float32)

    def compute(self, state, now):
        if self.reference_q is None:
            raise RuntimeError('local reference is not initialized')
        p = self.policy
        if self.return_at is not None:
            alpha = smoothstep((now-self.return_at)/self.cfg['return_s'])
            self.reference_q = (1-alpha)*self.return_from+alpha*self.saved_default
        w, x, y, z = state.quat_wxyz
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        live_pose = RobotPose(np.zeros(3), np.asarray([x,y,z,w]), now)
        target_pose = RobotPose(np.zeros(3), [0,0,np.sin(yaw/2),np.cos(yaw/2)], now)
        fk = p.kinematics[p.index]
        live = fk.forward(state.q_lab, live_pose)
        reference = fk.forward(self.reference_q, target_pose)
        self.history.update(state.quat_wxyz, state.gyro, state.q_lab, state.dq_lab, self.action)
        target, action = p.scalebfm.infer_batch(
            (self.history,), np.repeat(reference.body_pos_w[None,None], 6, axis=1),
            np.repeat(reference.body_quat_wxyz[None,None], 6, axis=1),
            live.body_pos_w[None], live.body_quat_wxyz[None],
            control_mode=p.control_mode, time_offsets=p.time_offsets)
        if not np.isfinite(target).all() or not np.isfinite(action).all():
            raise RuntimeError('non-finite local ScaleBFM output')
        self.action = action[0].copy()
        return PDCommand(target[0], p.kp, p.kd)


class CommandBlend:
    def __init__(self, seconds):
        self.seconds = seconds
        self.start = self.at = None

    def reset(self, command, now):
        self.start, self.at = command, now

    def apply(self, command, now):
        if self.start is None:
            return command
        alpha = smoothstep((now-self.at)/self.seconds)
        result = PDCommand((1-alpha)*self.start.target_pos+alpha*command.target_pos,
                           (1-alpha)*self.start.kp+alpha*command.kp,
                           (1-alpha)*self.start.kd+alpha*command.kd)
        if alpha == 1:
            self.start = None
        return result
