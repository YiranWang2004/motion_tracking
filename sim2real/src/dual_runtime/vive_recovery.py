"""Bounded Vive-only outage recovery; all times in robot A's wall-clock domain.

The ordinary team lease, fault checks and command watchdog remain mandatory.
Emergency local hold/fallback is immediate. Re-entering global tracking requires
a PREPARE/ACK/COMMIT exchange and a shared future interpolation interval.
"""
from dataclasses import replace
import math

import numpy as np

from .dual_pose_provider import DualPoseSnapshot


def recovery_settings(raw):
    cfg = dict(enabled=False, detect_after_s=.10, outage_timeout_s=1.,
               blend_s=.5, return_s=2., command_blend_s=.25,
               resume_delay_s=.20, valid_samples=3,
               max_position_jump_m=.35, max_orientation_jump_rad=.6)
    if raw is not None and (not isinstance(raw, dict) or set(raw)-set(cfg)):
        raise ValueError('unknown vive_recovery settings')
    cfg.update(raw or {})
    if not isinstance(cfg['enabled'], bool):
        raise ValueError('vive_recovery.enabled must be boolean')
    for key, value in cfg.items():
        if key != 'enabled' and (not isinstance(value, (int, float))
                                 or not math.isfinite(value) or value <= 0):
            raise ValueError(f'invalid vive_recovery.{key}')
    if not cfg['detect_after_s'] < cfg['outage_timeout_s']:
        raise ValueError('Vive detection must precede the outage deadline')
    if cfg['resume_delay_s'] < .1 or int(cfg['valid_samples']) != cfg['valid_samples']:
        raise ValueError('invalid Vive recovery handshake/sample settings')
    return cfg


def smoothstep(value):
    x = float(np.clip(value, 0., 1.))
    return x*x*(3.-2.*x)


def mix_snapshot(start, end, alpha):
    """SLERP orientations; keep real source timestamps, never refresh stale data."""
    result = []
    for a, b in zip((start.robot_a, start.robot_b, start.object),
                    (end.robot_a, end.robot_b, end.object)):
        qa, qb = a.quaternion_xyzw.astype(float), b.quaternion_xyzw.astype(float)
        dot = float(np.dot(qa, qb))
        if dot < 0:
            qb, dot = -qb, -dot
        angle = np.arccos(np.clip(dot, -1., 1.))
        if angle < 1e-5:
            q = (1-alpha)*qa + alpha*qb
        else:
            q = (np.sin((1-alpha)*angle)*qa + np.sin(alpha*angle)*qb)/np.sin(angle)
        result.append(replace(b, position_w=(1-alpha)*a.position_w+alpha*b.position_w,
                              quaternion_xyzw=q))
    return DualPoseSnapshot(*result)


class ViveRecovery:
    def __init__(self, side, settings):
        self.side, self.cfg = side, settings
        self.mode = 'normal'
        self.latest = self.applied = self.anchor = None
        self.sample_at = 0.
        self.healthy = False
        self.good_samples = 0
        self.loss = 0
        self.serial = 0
        self.plan = None
        self.ack = self.committed = 0
        self.fallback_at = None
        self.reject_reason = ''
        self.resumed_at = None
        self.frame = None
        self.last_sample_seq = None

    def observe(self, snapshot, now, leader_now, *, armed=True, sample_seq=None):
        """None means only a missing/stale sample, never malformed/future data."""
        self.resumed_at = None
        previous_healthy = self.healthy
        # A packet arriving on the first tick AFTER the deadline must not erase
        # the already-exceeded outage by replacing the last accepted timestamp.
        if (armed and self.latest is not None
                and now-self.latest.robot_a.stamp_s >= self.cfg['outage_timeout_s']):
            snapshot = None
        if snapshot is not None and self.mode != 'fallback':
            stamp = min(p.stamp_s for p in (snapshot.robot_a, snapshot.robot_b, snapshot.object))
            old_stamp = -math.inf if self.latest is None else self.latest.robot_a.stamp_s
            new_sequence = (sample_seq is None or self.last_sample_seq is None
                            or sample_seq > self.last_sample_seq)
            if stamp > old_stamp and new_sequence:
                if self.mode != 'normal' and self.anchor is not None:
                    for a, b in zip((self.anchor.robot_a, self.anchor.robot_b, self.anchor.object),
                                    (snapshot.robot_a, snapshot.robot_b, snapshot.object)):
                        distance = np.linalg.norm(a.position_w-b.position_w)
                        angle = 2*np.arccos(np.clip(abs(np.dot(a.quaternion_xyzw,
                                                               b.quaternion_xyzw)), 0., 1.))
                        if distance > self.cfg['max_position_jump_m'] or angle > self.cfg['max_orientation_jump_rad']:
                            self.reject_reason = 'vive_recovery_pose_jump'
                            break
                if not self.reject_reason:
                    self.latest = snapshot
                    self.sample_at = leader_now - max(0., now-stamp)
                    self.good_samples += 1
                    self.last_sample_seq = sample_seq
        age = math.inf if self.latest is None else now-self.latest.robot_a.stamp_s
        fresh = snapshot is not None and 0 <= age <= self.cfg['detect_after_s']
        if not fresh:
            self.good_samples = 0
        self.healthy = bool(fresh and not self.reject_reason and
                            (self.mode == 'normal' or self.good_samples >= self.cfg['valid_samples']))
        if previous_healthy and not self.healthy:
            self.loss += 1

    def status(self):
        return dict(mode=self.mode, healthy=self.healthy, sample_at=self.sample_at,
                    loss=self.loss, plan=self.plan, ack=self.ack, committed=self.committed,
                    fallback_at=self.fallback_at, rejected=bool(self.reject_reason))

    def update(self, peer, leader_now, next_frame):
        if not isinstance(peer, dict) or peer.get('mode') not in {'normal','grace','recovering','fallback'}:
            raise RuntimeError('vive_recovery_protocol_mismatch')
        for key in ('sample_at', 'loss', 'ack', 'committed'):
            if not isinstance(peer.get(key), (int, float)) or not math.isfinite(peer[key]):
                raise RuntimeError('invalid_vive_recovery_status')
        if not isinstance(peer.get('healthy'), bool) or not isinstance(peer.get('rejected'), bool):
            raise RuntimeError('invalid_vive_recovery_status')
        if peer.get('fallback_at') is not None and (
                not isinstance(peer['fallback_at'], (int,float)) or not math.isfinite(peer['fallback_at'])):
            raise RuntimeError('invalid_vive_recovery_status')
        remote_plan = peer.get('plan')
        if remote_plan is not None:
            if (not isinstance(remote_plan, dict) or set(remote_plan) != {'id','at','frame','losses','commit'}
                    or not isinstance(remote_plan['id'], int) or remote_plan['id'] < 1
                    or not isinstance(remote_plan['frame'], int) or remote_plan['frame'] < 0
                    or not isinstance(remote_plan['at'], (int,float)) or not math.isfinite(remote_plan['at'])
                    or not isinstance(remote_plan['commit'], bool)
                    or not isinstance(remote_plan['losses'], list) or len(remote_plan['losses']) != 2
                    or any(not isinstance(v, int) or v < 0 for v in remote_plan['losses'])):
                raise RuntimeError('invalid_vive_recovery_plan')
        if self.latest is None:
            raise RuntimeError('vive_recovery_requires_initial_pose')
        self.resumed_at = None
        if self.mode == 'fallback':
            return self.mode
        outage_at = min(self.sample_at, peer['sample_at']) + self.cfg['outage_timeout_s']
        if (self.reject_reason or peer.get('rejected') or peer['mode'] == 'fallback'
                or leader_now >= outage_at):
            at = peer.get('fallback_at')
            self.fallback_at = min(leader_now, outage_at) if at is None else float(at)
            self.mode = 'fallback'
            self.plan = None
            return self.mode
        unhealthy = not self.healthy or not peer['healthy']
        losses = [self.loss, peer['loss']] if self.side == 'a' else [peer['loss'], self.loss]
        completing_same_plan = (self.plan is not None and self.plan['losses'] == losses
                                and peer.get('plan') is not None
                                and peer['plan']['id'] == self.plan['id']
                                and peer['mode'] == 'recovering')
        if self.mode == 'normal' and not unhealthy and (peer['mode'] == 'normal' or completing_same_plan):
            self.applied = self.latest
            if peer['mode'] == 'normal':
                self.finish_resume()
            return self.mode
        if self.mode == 'normal':
            self.mode = 'grace'
            self.anchor = self.applied or self.latest
            self.good_samples = 0
        if unhealthy or (self.plan and self.plan['losses'] != losses):
            self.mode = 'grace'
            self.anchor = self.applied or self.latest
            self.plan = None
            self.ack = self.committed = 0
            return self.mode
        if self.side == 'a':
            if self.plan is None and peer['mode'] != 'normal':
                self.serial += 1
                self.plan = dict(id=self.serial, at=leader_now+self.cfg['resume_delay_s'],
                                 frame=int(next_frame), losses=losses, commit=False)
            if self.plan and peer['ack'] == self.plan['id']:
                self.plan['commit'] = True
        else:
            plan = peer.get('plan')
            if plan and plan['losses'] == losses:
                if self.plan is None or self.plan['id'] != plan['id']:
                    if plan['at']-leader_now < .05:
                        raise RuntimeError('late_vive_recovery_plan')
                    self.plan = dict(plan)
                elif any(self.plan[k] != plan[k] for k in ('at','frame','losses')):
                    raise RuntimeError('vive_recovery_plan_changed')
                self.plan['commit'] = plan['commit']
                self.ack = plan['id']
                if plan['commit']:
                    self.committed = plan['id']
        if self.plan and leader_now >= self.plan['at']:
            if not self.plan['commit'] or (self.side == 'a' and peer['committed'] != self.plan['id']):
                raise RuntimeError('vive_recovery_not_acknowledged')
            self.mode = 'recovering'
            self.frame = self.plan['frame']
            alpha = smoothstep((leader_now-self.plan['at'])/self.cfg['blend_s'])
            self.applied = mix_snapshot(self.anchor, self.latest, alpha)
            if leader_now >= self.plan['at']+self.cfg['blend_s']:
                self.resumed_at = self.plan['at']+self.cfg['blend_s']
                self.mode = 'normal'
                self.applied = self.latest
                # Keep plan/ack visible until both sides have crossed the boundary.
        return self.mode

    def finish_resume(self):
        self.plan = None
        self.ack = self.committed = 0

    def blend(self, leader_now):
        return 1. if self.mode == 'normal' else (0. if self.plan is None else
            smoothstep((leader_now-self.plan['at'])/self.cfg['blend_s']))
