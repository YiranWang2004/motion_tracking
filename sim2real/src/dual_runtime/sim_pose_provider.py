"""Atomic dual-robot pose provider populated by a simulation bridge state."""

from __future__ import annotations

import threading
import time
from typing import Any

from omnicontact.contracts import ObjectPose, RobotPose

from .dual_pose_provider import DualPoseSnapshot


class DualSimulationPoseProvider:
    """Convert one MuJoCo snapshot into the production dual-pose contract."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._snapshot: DualPoseSnapshot | None = None
        self._snapshot_id: int | None = None
        self.error: BaseException | None = None

    def start(self) -> dict[str, Any]:
        return {}

    def stop(self) -> None:
        with self._lock:
            self._snapshot = None
            self._snapshot_id = None
        self._ready.clear()

    def wait_until_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(max(0.0, float(timeout_s)))

    def get_snapshot(self) -> DualPoseSnapshot | None:
        with self._lock:
            return self._snapshot

    def ingest_bridge_state(self, data: dict[str, Any]) -> None:
        raw = data.get("dual_sim_pose")
        if not isinstance(raw, dict):
            raise RuntimeError("simulation bridge state is missing dual_sim_pose")
        try:
            snapshot_id = int(raw["snapshot_id"])
            robots = raw["robots"]
            obj = raw["object"]
            if not isinstance(robots, (list, tuple)) or len(robots) != 2:
                raise ValueError("robots must contain exactly A and B")
            if not isinstance(obj, dict):
                raise ValueError("object must be a mapping")
            stamp = time.monotonic()
            snapshot = DualPoseSnapshot(
                robot_a=RobotPose(
                    position_w=robots[0]["position_w"],
                    quaternion_xyzw=robots[0]["quaternion_xyzw"],
                    stamp_s=stamp,
                    confidence=1.0,
                ),
                robot_b=RobotPose(
                    position_w=robots[1]["position_w"],
                    quaternion_xyzw=robots[1]["quaternion_xyzw"],
                    stamp_s=stamp,
                    confidence=1.0,
                ),
                object=ObjectPose(
                    position_w=obj["position_w"],
                    quaternion_xyzw=obj["quaternion_xyzw"],
                    half_extents=obj["half_extents"],
                    stamp_s=stamp,
                    confidence=1.0,
                    linear_velocity_w=obj.get("linear_velocity_w"),
                    angular_velocity_w=obj.get("angular_velocity_w"),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed dual_sim_pose: {exc}") from exc
        with self._lock:
            if self._snapshot_id is not None and snapshot_id < self._snapshot_id:
                raise RuntimeError("dual_sim_pose snapshot_id moved backwards")
            self._snapshot = snapshot
            self._snapshot_id = snapshot_id
        self._ready.set()

