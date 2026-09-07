"""Atomic calibrated three-Tracker provider for dual-G1 deployment."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omnicontact.contracts import ObjectPose, RobotPose
from omnicontact.perception.openvr_tracker import OpenVRTrackerReader
from omnicontact.perception.vive_pose import (
    RigidTransform,
    _angular_velocity_world,
    sample_to_transform,
)

from .dual_pose_provider import DualPoseSnapshot


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualViveDeploymentConfig:
    robot_a_tracker_serial: str
    robot_b_tracker_serial: str
    object_tracker_serial: str
    world_from_steamvr: RigidTransform
    robot_a_tracker_to_pelvis: RigidTransform
    robot_b_tracker_to_pelvis: RigidTransform
    object_tracker_to_object: RigidTransform
    object_half_extents_m: np.ndarray
    calibration_confirmed: bool

    def __post_init__(self) -> None:
        serials = tuple(
            value.strip()
            for value in (
                self.robot_a_tracker_serial,
                self.robot_b_tracker_serial,
                self.object_tracker_serial,
            )
        )
        if any(not value or "REPLACE" in value.upper() for value in serials):
            raise ValueError("all three real Tracker serial numbers are required")
        if len(set(serials)) != 3:
            raise ValueError("robot A, robot B, and object must use different Trackers")
        if self.calibration_confirmed is not True:
            raise ValueError("calibration_confirmed must be true before deployment")
        for field, value in zip(
            (
                "robot_a_tracker_serial",
                "robot_b_tracker_serial",
                "object_tracker_serial",
            ),
            serials,
        ):
            object.__setattr__(self, field, value)
        extents = np.asarray(self.object_half_extents_m, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
            raise ValueError(
                "object_half_extents_m must contain positive finite values"
            )
        object.__setattr__(self, "object_half_extents_m", extents.copy())

    @classmethod
    def load(cls, path: str | Path, *, require_object: bool = True) -> DualViveDeploymentConfig:
        resolved = Path(path).expanduser().resolve()
        try:
            raw: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
            if not require_object:
                raw["object_tracker_serial"] = "__unused_object__"
                raw["object_tracker_to_object"] = {"position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
                raw["object_half_extents_m"] = [0.5, 0.15, 0.15]
            return cls(
                robot_a_tracker_serial=raw["robot_a_tracker_serial"],
                robot_b_tracker_serial=raw["robot_b_tracker_serial"],
                object_tracker_serial=raw["object_tracker_serial"],
                world_from_steamvr=RigidTransform.from_dict(
                    raw["world_from_steamvr"], "world_from_steamvr"
                ),
                robot_a_tracker_to_pelvis=RigidTransform.from_dict(
                    raw["robot_a_tracker_to_pelvis"], "robot_a_tracker_to_pelvis"
                ),
                robot_b_tracker_to_pelvis=RigidTransform.from_dict(
                    raw["robot_b_tracker_to_pelvis"], "robot_b_tracker_to_pelvis"
                ),
                object_tracker_to_object=RigidTransform.from_dict(
                    raw["object_tracker_to_object"], "object_tracker_to_object"
                ),
                object_half_extents_m=raw["object_half_extents_m"],
                calibration_confirmed=raw["calibration_confirmed"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid dual Vive config {resolved}: {exc}") from exc


class DualVivePoseProvider:
    """Read all three devices in one OpenVR call and publish one snapshot."""

    def __init__(
        self,
        config: DualViveDeploymentConfig,
        *,
        poll_hz: float = 100.0,
        require_object: bool = True,
        reader: OpenVRTrackerReader | None = None,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("poll_hz must be positive")
        self.require_object = require_object
        self.config = config
        self.poll_hz = float(poll_hz)
        self.serials = (
            config.robot_a_tracker_serial,
            config.robot_b_tracker_serial,
            config.object_tracker_serial,
        )
        if not require_object:
            self.serials = self.serials[:2]
        self.reader = reader or OpenVRTrackerReader(list(self.serials))
        self._lock = threading.Lock()
        self._snapshot: DualPoseSnapshot | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._last_object_transform: RigidTransform | None = None
        self._last_object_stamp: float | None = None
        self.valid_updates = 0
        self.invalid_updates = 0
        self._missing_since: dict[str, float] = {}
        self._missing_reported: dict[str, float] = {}

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
            target=self._run, name="dual-vive-pose-provider", daemon=True
        )
        self._thread.start()
        return devices

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.reader.stop()
        with self._lock:
            self._snapshot = None

    def wait_until_ready(self, timeout_s: float) -> bool:
        ready = self._ready_event.wait(max(0.0, timeout_s))
        if self._error is not None:
            raise RuntimeError(
                f"dual Vive provider failed: {self._error}"
            ) from self._error
        return ready and self.get_snapshot() is not None

    def get_snapshot(self) -> DualPoseSnapshot | None:
        with self._lock:
            return self._snapshot

    def update_once(self) -> bool:
        samples = self.reader.read_all(self.serials)
        now = time.monotonic()
        missing = [serial for serial in self.serials if samples.get(serial) is None]
        for serial in self.serials:
            if serial in missing:
                if serial not in self._missing_since:
                    self._missing_since[serial] = now
                    self._missing_reported[serial] = now
                    LOGGER.warning("Vive Tracker %s invalid; retaining last valid snapshot", serial)
                elif now - self._missing_reported[serial] >= 0.1:
                    LOGGER.warning("Vive Tracker %s still invalid for %.1f ms", serial,
                                   1000.0 * (now - self._missing_since[serial]))
                    self._missing_reported[serial] = now
            elif serial in self._missing_since:
                LOGGER.info("Vive Tracker %s recovered after %.1f ms", serial,
                            1000.0 * (now - self._missing_since.pop(serial)))
                self._missing_reported.pop(serial, None)
        if missing:
            self.invalid_updates += 1
            # Keep the complete atomic snapshot AND its original timestamps.
            # The coordinator enforces pose_timeout_s; partial updates must not
            # make old poses appear fresh. Startup still has no snapshot.
            self._last_object_transform = None
            self._last_object_stamp = None
            return False
        sample_a, sample_b = (samples[serial] for serial in self.serials[:2])
        stamp = time.monotonic()
        world_from_a = self.config.world_from_steamvr.compose(
            sample_to_transform(sample_a)
        ).compose(self.config.robot_a_tracker_to_pelvis)
        world_from_b = self.config.world_from_steamvr.compose(
            sample_to_transform(sample_b)
        ).compose(self.config.robot_b_tracker_to_pelvis)
        world_from_object = (
            self.config.world_from_steamvr.compose(
                sample_to_transform(samples[self.serials[2]])
            ).compose(self.config.object_tracker_to_object)
            if self.require_object else RigidTransform(np.zeros(3), np.array([0., 0., 0., 1.]))
        )
        linear_velocity = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        if (
            self._last_object_transform is not None
            and self._last_object_stamp is not None
        ):
            dt = stamp - self._last_object_stamp
            if 1.0e-4 <= dt <= 0.5:
                linear_velocity = (
                    (world_from_object.position - self._last_object_transform.position)
                    / dt
                ).astype(np.float32)
                angular_velocity = _angular_velocity_world(
                    self._last_object_transform.quaternion_xyzw,
                    world_from_object.quaternion_xyzw,
                    dt,
                )
        snapshot = DualPoseSnapshot(
            robot_a=RobotPose(
                world_from_a.position, world_from_a.quaternion_xyzw, stamp
            ),
            robot_b=RobotPose(
                world_from_b.position, world_from_b.quaternion_xyzw, stamp
            ),
            object=ObjectPose(
                world_from_object.position,
                world_from_object.quaternion_xyzw,
                self.config.object_half_extents_m,
                stamp,
                linear_velocity_w=linear_velocity,
                angular_velocity_w=angular_velocity,
            ),
        )
        with self._lock:
            self._snapshot = snapshot
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
        except Exception as exc:  # noqa: BLE001 - surfaced by wait_until_ready
            self._error = exc
            with self._lock:
                self._snapshot = None
            self._ready_event.set()
