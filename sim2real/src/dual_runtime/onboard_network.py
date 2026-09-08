"""Bounded UDP latest-value channels and measured remote clock offsets.

No socket waits occur on the control thread. A four-timestamp exchange maps
remote wall time to local time; half RTT is treated as timing uncertainty.
Endpoint addresses are configured explicitly. This is for a trusted robot LAN.
"""
from __future__ import annotations

from collections import deque
import json
import math
import socket
import threading
import time
import uuid

import numpy as np

from omnicontact.contracts import ObjectPose, RobotPose
from .dual_pose_provider import DualPoseSnapshot

PROTOCOL = "g1-onboard-v1"
MAX_PACKET = 8192


class ClockEstimate:
    def __init__(self, max_rtt_s=0.02):
        self.max_rtt_s = max_rtt_s
        self.samples = deque(maxlen=32)

    def observe(self, t0, t1, t2, t3, mono):
        if not all(math.isfinite(v) for v in (t0, t1, t2, t3, mono)):
            return
        rtt = (t3 - t0) - (t2 - t1)
        if not 0 <= rtt <= self.max_rtt_s or t2 < t1:
            return
        offset = ((t1 - t0) + (t2 - t3)) / 2
        self.samples.append((mono, rtt, offset))

    def estimate(self, mono):
        recent = [s for s in self.samples if 0 <= mono - s[0] < 2.0]
        if not recent:
            raise RuntimeError("clock_sync_not_ready_or_expired")
        _, rtt, offset = min(recent, key=lambda s: s[1])
        return offset, rtt / 2


class LatestChannel:
    def __init__(self, bind, remote, kind, *, max_rtt_s=0.02):
        self.remote = (socket.gethostbyname(remote[0]), int(remote[1]))
        self.kind = kind
        self.stream = uuid.uuid4().hex
        self.remote_stream = None
        self.seq = 0
        self.remote_seq = -1
        self.clock = ClockEstimate(max_rtt_s)
        self.latest = None
        self.error = None
        self.dropped = 0
        self._pending_pings = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(bind)
        self.sock.settimeout(0.05)
        self._wall_minus_mono = time.time() - time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"onboard-{kind}-receiver")
        self.thread.start()

    def _send(self, message):
        packet = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
        if len(packet) > MAX_PACKET:
            raise ValueError("onboard datagram too large")
        self.sock.sendto(packet, self.remote)

    def publish(self, payload):
        self.seq += 1
        self._send(dict(protocol=PROTOCOL, kind=self.kind, stream=self.stream,
                        seq=self.seq, sent=time.time(), payload=payload))

    def _run(self):
        next_ping = 0.
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_ping:
                    token = uuid.uuid4().hex
                    t0 = time.time()
                    with self._lock:
                        self._pending_pings = {k: v for k, v in self._pending_pings.items()
                                               if t0 - v < 2.}
                        self._pending_pings[token] = t0
                    self._send(dict(protocol=PROTOCOL, kind="ping", token=token, t0=t0))
                    next_ping = now + .2
                try:
                    data, sender = self.sock.recvfrom(MAX_PACKET + 1)
                except socket.timeout:
                    continue
                if sender != self.remote:
                    continue
                t3, mono = time.time(), time.monotonic()
                try:
                    msg = json.loads(data)
                    if len(data) > MAX_PACKET or msg.get("protocol") != PROTOCOL:
                        raise ValueError("invalid protocol")
                    if msg.get("kind") == "ping":
                        self._send(dict(protocol=PROTOCOL, kind="pong", token=msg["token"],
                                        t0=msg["t0"], t1=t3, t2=time.time()))
                    elif msg.get("kind") == "pong":
                        with self._lock:
                            t0 = self._pending_pings.pop(msg["token"], None)
                            if t0 is not None and t0 == msg["t0"]:
                                self.clock.observe(t0, float(msg["t1"]), float(msg["t2"]), t3, mono)
                    elif msg.get("kind") == self.kind:
                        seq, sent, stream = int(msg["seq"]), float(msg["sent"]), msg["stream"]
                        if not isinstance(stream, str) or not stream or not math.isfinite(sent):
                            raise ValueError("invalid envelope")
                        if not isinstance(msg["payload"], dict):
                            raise ValueError("invalid payload")
                        with self._lock:
                            if self.remote_stream is not None and stream != self.remote_stream:
                                self.error = "remote_stream_restarted"
                            elif seq > self.remote_seq:
                                self.remote_stream = stream
                                self.remote_seq = seq
                                self.latest = (msg, mono)
                except (ValueError, KeyError, TypeError, OverflowError, AttributeError):
                    self.dropped += 1
        except OSError as exc:
            if not self._stop.is_set():
                self.error = str(exc)

    def timing(self):
        if abs(time.time() - time.monotonic() - self._wall_minus_mono) > .02:
            raise RuntimeError("local_wall_clock_jump")
        with self._lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.clock.estimate(time.monotonic())

    def read(self, max_age_s):
        offset, uncertainty = self.timing()
        with self._lock:
            packet = self.latest
        if packet is None:
            raise RuntimeError(f"missing_{self.kind}")
        msg, arrival = packet
        age = time.time() - (msg["sent"] - offset)
        if age < -uncertainty - .002 or age + uncertainty > max_age_s:
            raise RuntimeError(f"stale_or_future_{self.kind}")
        if time.monotonic() - arrival > max_age_s:
            raise RuntimeError(f"missing_{self.kind}")
        return msg["payload"], offset, uncertainty

    def close(self):
        self._stop.set()
        self.thread.join(timeout=.2)
        self.sock.close()


def pose_payload(snapshot, calibration_id, seq):
    """Old samples keep their acquisition time even when retransmitted."""
    age = time.monotonic() - min(snapshot.robot_a.stamp_s,
                                  snapshot.robot_b.stamp_s, snapshot.object.stamp_s)
    return dict(
        calibration_id=calibration_id, snapshot_seq=seq,
        captured=time.time() - age,
        poses=[[*p.position_w.tolist(), *p.quaternion_xyzw.tolist()]
               for p in (snapshot.robot_a, snapshot.robot_b, snapshot.object)],
        half_extents=snapshot.object.half_extents.tolist(),
    )


def decode_pose(payload, *, calibration_id, offset, uncertainty, max_age_s,
                wall=None, mono=None):
    wall = time.time() if wall is None else wall
    mono = time.monotonic() if mono is None else mono
    if payload.get("calibration_id") != calibration_id:
        raise RuntimeError("vive_calibration_mismatch")
    age = wall - (float(payload["captured"]) - offset)
    if not math.isfinite(age) or age < -uncertainty-.002 or age+uncertainty > max_age_s:
        raise RuntimeError("stale_or_future_vive_capture")
    poses = np.asarray(payload["poses"], dtype=np.float32)
    extents = np.asarray(payload["half_extents"], dtype=np.float32)
    if poses.shape != (3, 7) or extents.shape != (3,) or not np.isfinite(poses).all() or not np.isfinite(extents).all() or np.any(extents <= 0):
        raise ValueError("invalid network poses")
    norms = np.linalg.norm(poses[:, 3:], axis=-1)
    if np.any(np.abs(norms-1) > .01):
        raise ValueError("invalid network pose quaternion")
    poses[:, 3:] /= norms[:, None]
    stamp = mono - max(0., age + uncertainty)
    return DualPoseSnapshot(
        RobotPose(poses[0, :3], poses[0, 3:], stamp),
        RobotPose(poses[1, :3], poses[1, 3:], stamp),
        ObjectPose(poses[2, :3], poses[2, 3:], extents, stamp),
    )
