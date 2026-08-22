"""Calibrated two-Tracker pose provider for OmniContact deployment."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omnicontact.perception.object_pose import ExternalObjectPoseProvider
from omnicontact.perception.openvr_tracker import OpenVRTrackerReader, ViveSample
from omnicontact.contracts import ObjectPose, RobotPose


def _normalize_quaternion(quaternion_xyzw: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("quaternion must be non-zero")
    return quaternion / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_xyz = quaternion[:3]
    uv = np.cross(q_xyz, vector)
    uuv = np.cross(q_xyz, uv)
    return vector + 2.0 * (quaternion[3] * uv + uuv)


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform ``parent_from_child`` with an xyzw quaternion."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("transform position contains non-finite values")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _normalize_quaternion(self.quaternion_xyzw),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any], field_name: str) -> "RigidTransform":
        try:
            return cls(
                position=value["position_m"],
                quaternion_xyzw=value["quaternion_xyzw"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid transform {field_name}: {exc}") from exc

    def compose(self, child_from_grandchild: "RigidTransform") -> "RigidTransform":
        position = self.position + _quat_rotate(
            self.quaternion_xyzw, child_from_grandchild.position
        )
        quaternion = _quat_multiply(
            self.quaternion_xyzw, child_from_grandchild.quaternion_xyzw
        )
        return RigidTransform(position, quaternion)


@dataclass(frozen=True)
class ViveDeploymentConfig:
    """All measured values needed to turn Tracker poses into task poses.

    Transform convention:

    ``world_from_steamvr`` is ^W T_S.
    ``robot_tracker_to_pelvis`` is ^T_robot T_pelvis.
    ``object_tracker_to_object`` is ^T_object T_object.
    """

    robot_tracker_serial: str
    object_tracker_serial: str
    world_from_steamvr: RigidTransform
    robot_tracker_to_pelvis: RigidTransform
    object_tracker_to_object: RigidTransform
    object_half_extents_m: np.ndarray
    goal_position_w: np.ndarray
    calibration_confirmed: bool

    def __post_init__(self) -> None:
        robot_serial = self.robot_tracker_serial.strip()
        object_serial = self.object_tracker_serial.strip()
        if not robot_serial or not object_serial:
            raise ValueError("both Tracker serial numbers are required")
        if robot_serial == object_serial:
            raise ValueError("robot and object Tracker serial numbers must differ")
        if "REPLACE" in robot_serial.upper() or "REPLACE" in object_serial.upper():
            raise ValueError("replace example Tracker serial numbers before deployment")
        if self.calibration_confirmed is not True:
            raise ValueError(
                "calibration_confirmed must be true after all transforms, dimensions, and goal are measured"
            )
        object.__setattr__(self, "robot_tracker_serial", robot_serial)
        object.__setattr__(self, "object_tracker_serial", object_serial)

        half_extents = np.asarray(self.object_half_extents_m, dtype=np.float64).reshape(3)
        goal = np.asarray(self.goal_position_w, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(half_extents)) or np.any(half_extents <= 0.0):
            raise ValueError("object_half_extents_m must contain three positive values")
        if not np.all(np.isfinite(goal)):
            raise ValueError("goal_position_w contains non-finite values")
        object.__setattr__(self, "object_half_extents_m", half_extents.copy())
        object.__setattr__(self, "goal_position_w", goal.copy())

    @classmethod
    def load(cls, path: str | Path) -> "ViveDeploymentConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read Vive deployment config {config_path}: {exc}") from exc
        try:
            return cls(
                robot_tracker_serial=raw["robot_tracker_serial"],
                object_tracker_serial=raw["object_tracker_serial"],
                world_from_steamvr=RigidTransform.from_dict(
                    raw["world_from_steamvr"], "world_from_steamvr"
                ),
                robot_tracker_to_pelvis=RigidTransform.from_dict(
                    raw["robot_tracker_to_pelvis"], "robot_tracker_to_pelvis"
                ),
                object_tracker_to_object=RigidTransform.from_dict(
                    raw["object_tracker_to_object"], "object_tracker_to_object"
                ),
                object_half_extents_m=raw["object_half_extents_m"],
                goal_position_w=raw["goal_position_w"],
                calibration_confirmed=raw["calibration_confirmed"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Vive deployment config {config_path}: {exc}") from exc


def sample_to_transform(sample: ViveSample) -> RigidTransform:
    return RigidTransform(
        position=np.array(
            [sample.line_x_m, sample.line_y_m, sample.line_z_m],
            dtype=np.float64,
        ),
        quaternion_xyzw=np.array(
            [sample.qx, sample.qy, sample.qz, sample.qw],
            dtype=np.float64,
        ),
    )


def _angular_velocity_world(
    previous_quaternion: np.ndarray,
    current_quaternion: np.ndarray,
    dt: float,
) -> np.ndarray:
    previous_inverse = previous_quaternion.copy()
    previous_inverse[:3] *= -1.0
    delta = _normalize_quaternion(
        _quat_multiply(current_quaternion, previous_inverse)
    )
    if delta[3] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[:3]))
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(delta[3], -1.0, 1.0)))
    return (delta[:3] / vector_norm * (angle / dt)).astype(np.float32)


class VivePoseProvider(ExternalObjectPoseProvider):
    """Background OpenVR adapter publishing a calibrated robot/object pair."""

    def __init__(
        self,
        config: ViveDeploymentConfig,
        *,
        poll_hz: float = 100.0,
        reader: OpenVRTrackerReader | None = None,
    ) -> None:
        super().__init__()
        if poll_hz <= 0.0:
            raise ValueError("poll_hz must be positive")
        self.config = config
        self.poll_hz = float(poll_hz)
        self.reader = reader or OpenVRTrackerReader(
            [config.robot_tracker_serial, config.object_tracker_serial]
        )
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._last_object_transform: RigidTransform | None = None
        self._last_object_stamp: float | None = None
        self.valid_updates = 0
        self.invalid_updates = 0

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> dict[str, int]:
        if self._thread is not None and self._thread.is_alive():
            return self.reader.serial_to_index
        devices = self.reader.start()
        self._stop_event.clear()
        self._ready_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="vive-pose-provider",
            daemon=True,
        )
        self._thread.start()
        return devices

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.reader.stop()
        self.clear()

    def wait_until_ready(self, timeout_s: float) -> bool:
        ready = self._ready_event.wait(timeout=max(0.0, timeout_s))
        if self._error is not None:
            raise RuntimeError(f"Vive pose provider failed: {self._error}") from self._error
        return ready

    def update_once(self) -> bool:
        serials = (
            self.config.robot_tracker_serial,
            self.config.object_tracker_serial,
        )
        samples = self.reader.read_all(serials)
        robot_sample = samples.get(serials[0])
        object_sample = samples.get(serials[1])
        if robot_sample is None or object_sample is None:
            self.invalid_updates += 1
            self._last_object_transform = None
            self._last_object_stamp = None
            self.clear()
            return False

        stamp = time.monotonic()
        world_from_robot_tracker = self.config.world_from_steamvr.compose(
            sample_to_transform(robot_sample)
        )
        world_from_object_tracker = self.config.world_from_steamvr.compose(
            sample_to_transform(object_sample)
        )
        world_from_pelvis = world_from_robot_tracker.compose(
            self.config.robot_tracker_to_pelvis
        )
        world_from_object = world_from_object_tracker.compose(
            self.config.object_tracker_to_object
        )

        linear_velocity = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        if self._last_object_transform is not None and self._last_object_stamp is not None:
            dt = stamp - self._last_object_stamp
            if 1e-4 <= dt <= 0.5:
                linear_velocity = (
                    (world_from_object.position - self._last_object_transform.position) / dt
                ).astype(np.float32)
                angular_velocity = _angular_velocity_world(
                    self._last_object_transform.quaternion_xyzw,
                    world_from_object.quaternion_xyzw,
                    dt,
                )

        robot_pose = RobotPose(
            position_w=world_from_pelvis.position,
            quaternion_xyzw=world_from_pelvis.quaternion_xyzw,
            stamp_s=stamp,
            confidence=1.0,
        )
        object_pose = ObjectPose(
            position_w=world_from_object.position,
            quaternion_xyzw=world_from_object.quaternion_xyzw,
            half_extents=self.config.object_half_extents_m,
            stamp_s=stamp,
            confidence=1.0,
            linear_velocity_w=linear_velocity,
            angular_velocity_w=angular_velocity,
        )
        self.publish_pair(robot_pose, object_pose)
        self._last_object_transform = world_from_object
        self._last_object_stamp = stamp
        self.valid_updates += 1
        self._ready_event.set()
        return True

    def _run(self) -> None:
        interval = 1.0 / self.poll_hz
        next_tick = time.monotonic()
        last_refresh = next_tick
        try:
            while not self._stop_event.is_set():
                valid = self.update_once()
                now = time.monotonic()
                if not valid and now - last_refresh >= 1.0:
                    self.reader.refresh_devices()
                    last_refresh = now
                next_tick += interval
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    self._stop_event.wait(delay)
                elif delay < -interval:
                    next_tick = time.monotonic()
        except BaseException as exc:  # surfaced by wait_until_ready/status checks
            self._error = exc
            self.clear()
            self._ready_event.set()


__all__ = [
    "RigidTransform",
    "ViveDeploymentConfig",
    "VivePoseProvider",
    "sample_to_transform",
]
