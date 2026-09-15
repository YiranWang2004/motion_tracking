"""Timestamped Pico frames and a delayed, causal six-frame reference window."""

import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
import zmq
from omnicontact.contracts import RobotPose
from scalebfm.constants import POLICY_JOINT_NAMES
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Frame:
    stamp: float
    position: np.ndarray
    quaternion: np.ndarray  # WXYZ
    joints: np.ndarray  # policy order


def decode_frame(header, payload, stamp):
    if header.get("protocol_version") != 2:
        raise ValueError("Pico server requires protocol v2; restart the updated server")
    names = header["joint_names"]
    if len(names) != 29 or set(names) != set(POLICY_JOINT_NAMES):
        raise ValueError("Pico joint names do not match G1")
    if header["num_frames"] != 1 or header["qpos_size"] != 36:
        raise ValueError("expected one G1 qpos frame")
    q = np.frombuffer(payload, dtype=np.float32).copy()
    if q.shape != (36,) or not np.isfinite(q).all() or np.linalg.norm(q[3:7]) < 1e-6:
        raise ValueError("invalid Pico qpos")
    return Frame(
        stamp,
        q[:3],
        q[3:7] / np.linalg.norm(q[3:7]),
        q[7:][[names.index(n) for n in POLICY_JOINT_NAMES]],
    )


class PicoSource:
    """Single consumer of the existing PUSH/PULL service; never refresh stale poses.

    Freshness uses sender age plus request round-trip, avoiding cross-host clock
    assumptions. The sender monotonic timestamp is used only for sample deltas.
    """

    def __init__(self, host="127.0.0.1", timeout=0.25):
        self.timeout = timeout
        self.context = zmq.Context()
        self.sockets = []
        for kind, port in ((zmq.PUSH, 28701), (zmq.PULL, 28702), (zmq.PULL, 28703)):
            s = self.context.socket(kind)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.SNDHWM, 2)
            s.setsockopt(zmq.RCVHWM, 32)
            s.connect(f"tcp://{host}:{port}")
            self.sockets.append(s)
        self.req, self.rep, self.ctrl = self.sockets
        self.pending = {}
        self.counter = 0
        self.session_id = uuid.uuid4().hex
        self.enabled = False
        self.generation = 0
        self.buttons = None
        self.last_sender = None
        self.origin_sender = None
        self.origin_local = None
        self.last_valid = None
        self.frames = deque(maxlen=150)

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        for _ in range(32):
            try:
                self.ctrl.recv_json(zmq.NOBLOCK)
            except zmq.Again:
                break
            # Legacy stream has no correlated freshness; controls are taken
            # from the v2 reply below. Drain it only to bound server queues.
        for _ in range(32):
            try:
                h, payload = self.rep.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            import json

            header = json.loads(h)
            if header.get("protocol_version") != 2:
                raise ValueError(
                    "Pico server requires protocol v2; restart the updated server"
                )
            sent = self.pending.pop(header.get("request_id"), None)
            if sent is None:
                continue
            age = float(header.get("sample_age_s", float("inf"))) + now - sent
            if not np.isfinite(age) or not 0 <= age <= self.timeout:
                continue
            control_age = header.get("control_age_s")
            if (
                control_age is None
                or not 0 <= float(control_age) + now - sent <= self.timeout
            ):
                continue
            b = header.get("controller_buttons", {})
            start, stop = bool(b.get("right_key_one")), bool(b.get("left_key_one"))
            if stop:
                self.enabled = False
            elif self.buttons is not None and start and not self.buttons[0]:
                self.enabled = True
                self.generation += 1
            self.buttons = start, stop
            sender = int(header["sample_time_ns"])
            if self.last_sender is not None and sender <= self.last_sender:
                continue
            if self.origin_sender is None:
                self.origin_sender, self.origin_local = sender, now - age
            stamp = self.origin_local + (sender - self.origin_sender) * 1e-9
            frame = decode_frame(header, payload, stamp)
            self.last_sender, self.last_valid = sender, now - age
            self.frames.append(frame)
        self.pending = {k: v for k, v in self.pending.items() if now - v < self.timeout}
        if len(self.pending) < 2:
            self.counter += 1
            request_id = f"{self.session_id}:{self.counter}"
            try:
                self.req.send_json(
                    {"request_id": request_id, "start": False}, zmq.NOBLOCK
                )
                self.pending[request_id] = now
            except zmq.Again:
                pass

    def fresh(self, now):
        return (
            self.last_valid is not None and 0 <= now - self.last_valid <= self.timeout
        )

    def close(self):
        for s in self.sockets:
            s.close()
        self.context.term()


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def yaw(q):
    r = rotation(q).as_matrix()
    return np.arctan2(r[1, 0], r[0, 0])


class ReferenceWindow:
    def __init__(self, fk, offsets=(0, 1, 2, 3, 4, 5), transition_s=0.5):
        self.fk = fk
        self.offsets = np.asarray(offsets, dtype=np.int64)
        if (
            self.offsets.shape != (6,)
            or not np.array_equal(self.offsets, np.asarray(offsets))
            or not np.array_equal(self.offsets[:5], np.arange(5))
            or not 5 <= self.offsets[5] <= 33
        ):
            raise ValueError("reference offsets must be [0,1,2,3,4,k], 5<=k<=33")
        self.transition_s = transition_s
        if not np.isfinite(transition_s) or transition_s <= 0:
            raise ValueError("transition_s must be finite and positive")
        self.anchor = None

    def reset(self, frame, pose, joints, now):
        self.source_origin = frame.position.copy()
        self.delta = Rotation.from_euler(
            "z", yaw(pose.quaternion_xyzw[[3, 0, 1, 2]]) - yaw(frame.quaternion)
        )
        self.anchor = pose
        self.start_q = np.asarray(joints).copy()
        self.started = now

    def build(self, frames, now):
        frames = list(frames)
        if self.anchor is None or len(frames) < 2:
            raise RuntimeError("reference window not initialized")
        times = np.array([f.stamp for f in frames])
        # Delayed playback: all six samples already exist; no invented future.
        start = times[-1] - self.offsets[-1] / 50.0
        if start < times[0]:
            raise RuntimeError("reference window warming up")
        sample_times = np.clip(start + self.offsets / 50.0, times[0], times[-1])
        positions = np.array([f.position for f in frames])
        joints = np.array([f.joints for f in frames])
        quats = Slerp(
            times,
            Rotation.from_quat(np.array([f.quaternion[[1, 2, 3, 0]] for f in frames])),
        )(sample_times)
        blend = np.clip((now - self.started) / self.transition_s, 0, 1)
        p_out, q_out = [], []
        for i, t in enumerate(sample_times):
            p = np.array([np.interp(t, times, positions[:, k]) for k in range(3)])
            p = self.anchor.position_w + self.delta.apply(p - self.source_origin)
            q = (self.delta * quats[i]).as_quat()
            j = np.array([np.interp(t, times, joints[:, k]) for k in range(29)])
            p = self.anchor.position_w + blend * (p - self.anchor.position_w)
            q = Slerp([0, 1], Rotation.from_quat([self.anchor.quaternion_xyzw, q]))(
                [blend]
            ).as_quat()[0]
            j = self.start_q + blend * (j - self.start_q)
            live = self.fk.forward(j, RobotPose(p, q, now))
            p_out.append(live.body_pos_w)
            q_out.append(live.body_quat_wxyz)
        return np.array(p_out)[None], np.array(q_out)[None]
