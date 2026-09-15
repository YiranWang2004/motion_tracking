"""Single-pelvis localization. Local odometry is an estimate, never mocap."""

import json
import time

import numpy as np
import zmq
from omnicontact.contracts import RobotPose
from omnicontact.perception.openvr_tracker import OpenVRTrackerReader
from scipy.spatial.transform import Rotation

from .reference import rotation, yaw


class LocalOdometry:
    """IMU orientation + stance-foot FK translation on a flat floor.

    No force sensors: contact is a geometric hypothesis. Sliding, flight, stairs
    and carrying the robot invalidate it; this is NOT a global position sensor.
    """

    def __init__(self, fk, sole_height=0.03, contact_height=0.04, max_speed=2.0):
        if not all(
            np.isfinite(v) and v > 0 for v in (sole_height, contact_height, max_speed)
        ):
            raise ValueError("local odometry parameters must be finite and positive")
        self.fk = fk
        self.sole_height, self.contact_height, self.max_speed = (
            sole_height,
            contact_height,
            max_speed,
        )
        self.origin = None
        self.anchors = {}
        self.last = None

    def update(self, state, now):
        if self.origin is None:
            self.origin = Rotation.from_euler("z", -yaw(state.quat_wxyz))
        quat = (self.origin * rotation(state.quat_wxyz)).as_quat()
        feet = self.fk.forward(
            state.q_lab, RobotPose(np.zeros(3), quat, now)
        ).body_pos_w[[3, 6]]
        if self.last is None:
            self.position = np.array([0.0, 0.0, self.sole_height - feet[:, 2].min()])
        dt = 0.02 if self.last is None else now - self.last
        if not 0 < dt <= 0.25:
            raise RuntimeError("local odometry state gap")
        predicted = feet + self.position
        contact = np.flatnonzero(
            np.abs(predicted[:, 2] - self.sole_height) <= self.contact_height
        )
        if not len(contact):
            raise RuntimeError(
                "local odometry lost floor contact; no automatic fallback"
            )
        # Retain fixed world anchors only for feet still on the floor.
        self.anchors = {
            int(i): self.anchors.get(int(i), predicted[i].copy()) for i in contact
        }
        position = np.mean([self.anchors[int(i)] - feet[i] for i in contact], axis=0)
        if np.linalg.norm(position - self.position) / dt > self.max_speed:
            raise RuntimeError("local odometry discontinuity")
        self.position, self.last = position, now
        return RobotPose(position, quat, now, confidence=0.5)

    def close(self):
        pass


class ViveReader:
    """Read exactly one tracker with T_world_steamvr * T_steamvr_tracker * T_tracker_pelvis."""

    def __init__(self, path, robot="a"):
        with open(path) as f:
            cfg = json.load(f)
        if not cfg.get("calibration_confirmed"):
            raise ValueError("Vive calibration must be confirmed")
        self.serial = cfg[f"robot_{robot}_tracker_serial"]
        self.world = cfg["world_from_steamvr"]
        self.mount = cfg[f"robot_{robot}_tracker_to_pelvis"]
        # Validate transforms before opening OpenVR.
        for t in (self.world, self.mount):
            RobotPose(t["position_m"], t["quaternion_xyzw"], 0.0)
        self.reader = OpenVRTrackerReader([self.serial])
        self.reader.start()

    def read(self):
        sample = self.reader.read_all([self.serial]).get(self.serial)
        if sample is None:
            return None
        wr = Rotation.from_quat(self.world["quaternion_xyzw"])
        tr = Rotation.from_quat([sample.qx, sample.qy, sample.qz, sample.qw])
        p = np.asarray(self.world["position_m"]) + wr.apply(
            np.array([sample.line_x_m, sample.line_y_m, sample.line_z_m])
            + tr.apply(self.mount["position_m"])
        )
        q = (wr * tr * Rotation.from_quat(self.mount["quaternion_xyzw"])).as_quat()
        return RobotPose(p, q, time.monotonic())

    def close(self):
        self.reader.stop()


class ViveSource:
    """Correlated request/reply: freshness bounded by local RTT, no clock sync."""

    def __init__(self, endpoint, timeout=0.1):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.RCVHWM, 2)
        self.socket.connect(endpoint)
        self.pending = {}
        self.counter = 0
        self.timeout = timeout
        self.latest = None
        self.alignment = None

    def poll(self, now):
        for _ in range(4):
            try:
                reply = self.socket.recv_json(zmq.NOBLOCK)
            except zmq.Again:
                break
            sent = self.pending.pop(reply.get("id"), None)
            if (
                sent is not None
                and 0 <= now - sent <= self.timeout
                and reply.get("valid")
            ):
                pose = RobotPose(reply["p"], reply["q"], sent)
                if self.latest is None or pose.stamp_s > self.latest.stamp_s:
                    self.latest = pose
        self.pending = {k: v for k, v in self.pending.items() if now - v < self.timeout}
        if len(self.pending) < 2:
            self.counter += 1
            try:
                self.socket.send_json({"id": self.counter}, zmq.NOBLOCK)
                self.pending[self.counter] = now
            except zmq.Again:
                pass

    def update(self, state, now):
        self.poll(now)
        if self.latest is None or not 0 <= now - self.latest.stamp_s <= self.timeout:
            raise RuntimeError("Vive pelvis unavailable/stale")
        q = self.latest.quaternion_xyzw[[3, 0, 1, 2]]
        if self.alignment is None:
            self.alignment = Rotation.from_euler("z", yaw(q) - yaw(state.quat_wxyz))
        imu = self.alignment * rotation(state.quat_wxyz)
        if (imu.inv() * rotation(q)).magnitude() > 0.35:
            raise RuntimeError(
                "Vive/IMU orientation mismatch: check mounting and calibration"
            )
        # IMU roll/pitch and aligned heading are also used by policy history.
        return RobotPose(self.latest.position_w, imu.as_quat(), self.latest.stamp_s)

    def close(self):
        self.socket.close()
        self.context.term()


def serve_vive(config, robot, endpoint):
    reader = ViveReader(config, robot)
    context = zmq.Context()
    sock = context.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.bind(endpoint)
    try:
        while True:
            if not sock.poll(100):
                continue
            identity, message = sock.recv_multipart()
            request = json.loads(message)
            pose = reader.read()  # Fresh read AFTER request, never resend cached pose.
            reply = {"id": request["id"], "valid": pose is not None}
            if pose is not None:
                reply.update(
                    p=pose.position_w.tolist(), q=pose.quaternion_xyzw.tolist()
                )
            try:
                sock.send_multipart([identity, json.dumps(reply).encode()], zmq.NOBLOCK)
            except zmq.Again:
                pass
    finally:
        reader.close()
        sock.close()
        context.term()
