"""Authenticated UDP transport for calibrated OmniContact pose pairs."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import threading
import time
import uuid
from typing import Any

import numpy as np

from omnicontact.perception.object_pose import ExternalObjectPoseProvider
from omnicontact.contracts import ObjectPose, RobotPose


PROTOCOL_NAME = "robojudo.omnicontact.pose"
PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 16_384


def _pose_payload(robot_pose: RobotPose, object_pose: ObjectPose) -> dict[str, Any]:
    return {
        "robot": {
            "position_w": robot_pose.position_w.tolist(),
            "quaternion_xyzw": robot_pose.quaternion_xyzw.tolist(),
            "confidence": float(robot_pose.confidence),
        },
        "object": {
            "position_w": object_pose.position_w.tolist(),
            "quaternion_xyzw": object_pose.quaternion_xyzw.tolist(),
            "half_extents": object_pose.half_extents.tolist(),
            "linear_velocity_w": (
                None
                if object_pose.linear_velocity_w is None
                else np.asarray(object_pose.linear_velocity_w, dtype=float).tolist()
            ),
            "angular_velocity_w": (
                None
                if object_pose.angular_velocity_w is None
                else np.asarray(object_pose.angular_velocity_w, dtype=float).tolist()
            ),
            "confidence": float(object_pose.confidence),
        },
    }


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_pose_packet(
    robot_pose: RobotPose,
    object_pose: ObjectPose,
    *,
    sequence: int,
    stream_id: str,
    token: str,
) -> bytes:
    if not token:
        raise ValueError("UDP token must not be empty")
    payload = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "stream_id": stream_id,
        "sequence": int(sequence),
        "source_wall_time_ns": time.time_ns(),
        **_pose_payload(robot_pose, object_pose),
    }
    payload_bytes = _canonical_json(payload)
    signature = hmac.new(token.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    packet = _canonical_json({"payload": payload, "hmac_sha256": signature})
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError(f"pose packet is too large: {len(packet)} bytes")
    return packet


def decode_pose_packet(packet: bytes, token: str) -> tuple[str, int, RobotPose, ObjectPose]:
    if not token:
        raise ValueError("UDP token must not be empty")
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError("pose packet exceeds maximum size")
    try:
        envelope = json.loads(packet.decode("utf-8"))
        payload = envelope["payload"]
        supplied_signature = envelope["hmac_sha256"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed pose packet: {exc}") from exc

    expected_signature = hmac.new(
        token.encode("utf-8"), _canonical_json(payload), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(str(supplied_signature), expected_signature):
        raise ValueError("pose packet authentication failed")
    if payload.get("protocol") != PROTOCOL_NAME or payload.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported pose packet protocol/version")

    try:
        stream_id = str(payload["stream_id"])
        uuid.UUID(stream_id)
        sequence = int(payload["sequence"])
        if sequence < 0:
            raise ValueError("negative sequence")
        robot = payload["robot"]
        obj = payload["object"]
        # Receiver freshness is based on local arrival time, never an
        # unsynchronised sender monotonic clock.
        stamp = time.monotonic()
        robot_pose = RobotPose(
            position_w=robot["position_w"],
            quaternion_xyzw=robot["quaternion_xyzw"],
            confidence=float(robot["confidence"]),
            stamp_s=stamp,
        )
        object_pose = ObjectPose(
            position_w=obj["position_w"],
            quaternion_xyzw=obj["quaternion_xyzw"],
            half_extents=obj["half_extents"],
            confidence=float(obj["confidence"]),
            stamp_s=stamp,
            linear_velocity_w=(
                None
                if obj.get("linear_velocity_w") is None
                else np.asarray(obj["linear_velocity_w"], dtype=np.float32).reshape(3)
            ),
            angular_velocity_w=(
                None
                if obj.get("angular_velocity_w") is None
                else np.asarray(obj["angular_velocity_w"], dtype=np.float32).reshape(3)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid pose packet fields: {exc}") from exc
    return stream_id, sequence, robot_pose, object_pose


class PoseUdpPublisher:
    def __init__(
        self,
        target_host: str,
        port: int,
        token: str,
        *,
        bind_ip: str | None = None,
    ) -> None:
        if not 1 <= int(port) <= 65535:
            raise ValueError("UDP port must be in [1, 65535]")
        if not token:
            raise ValueError("UDP token must not be empty")
        self.target = (socket.gethostbyname(target_host), int(port))
        self.token = token
        self.stream_id = str(uuid.uuid4())
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if bind_ip is not None:
            self.socket.bind((bind_ip, 0))

    def send(self, robot_pose: RobotPose, object_pose: ObjectPose) -> int:
        packet = encode_pose_packet(
            robot_pose,
            object_pose,
            sequence=self.sequence,
            stream_id=self.stream_id,
            token=self.token,
        )
        sent = self.socket.sendto(packet, self.target)
        self.sequence += 1
        return sent

    def close(self) -> None:
        self.socket.close()


class UdpPoseReceiverProvider(ExternalObjectPoseProvider):
    """Receive authenticated calibrated poses on the policy host."""

    def __init__(
        self,
        bind_ip: str,
        port: int,
        token: str,
        *,
        allowed_sender_ip: str | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= int(port) <= 65535:
            raise ValueError("UDP port must be in [1, 65535]")
        if not token:
            raise ValueError("UDP token must not be empty")
        self.bind_ip = bind_ip
        self.port = int(port)
        self.token = token
        self.allowed_sender_ip = (
            None if allowed_sender_ip is None else socket.gethostbyname(allowed_sender_ip)
        )
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error: BaseException | None = None
        self._stream_id: str | None = None
        self._last_sequence = -1
        self.last_sender: tuple[str, int] | None = None
        self.valid_packets = 0
        self.invalid_packets = 0

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.settimeout(0.2)
        sock.bind((self.bind_ip, self.port))
        self._socket = sock
        self._stop_event.clear()
        self._ready_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="omnicontact-pose-udp",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._socket = None
        self._thread = None
        self.clear()

    def wait_until_ready(self, timeout_s: float) -> bool:
        ready = self._ready_event.wait(timeout=max(0.0, timeout_s))
        if self._error is not None:
            raise RuntimeError(f"UDP pose receiver failed: {self._error}") from self._error
        return ready

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    assert self._socket is not None
                    packet, sender = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                if self.allowed_sender_ip is not None and sender[0] != self.allowed_sender_ip:
                    self.invalid_packets += 1
                    continue
                try:
                    stream_id, sequence, robot_pose, object_pose = decode_pose_packet(
                        packet, self.token
                    )
                except ValueError:
                    self.invalid_packets += 1
                    continue

                if stream_id != self._stream_id:
                    self._stream_id = stream_id
                    self._last_sequence = -1
                if sequence <= self._last_sequence:
                    self.invalid_packets += 1
                    continue
                self._last_sequence = sequence
                self.last_sender = sender
                self.publish_pair(robot_pose, object_pose)
                self.valid_packets += 1
                self._ready_event.set()
        except BaseException as exc:
            self._error = exc
            self.clear()
            self._ready_event.set()


__all__ = [
    "PoseUdpPublisher",
    "UdpPoseReceiverProvider",
    "decode_pose_packet",
    "encode_pose_packet",
]

