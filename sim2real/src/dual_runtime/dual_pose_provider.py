"""Contract for the future three-Tracker Vive provider.

The independent routing test does not require Vive.  The production provider
should publish one atomic snapshot containing robot A, robot B, and the shared
object pose from one OpenVR read cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from omnicontact.contracts import ObjectPose, RobotPose


@dataclass(frozen=True)
class DualPoseSnapshot:
    robot_a: RobotPose
    robot_b: RobotPose
    object: ObjectPose


class DualPoseProvider(Protocol):
    def get_snapshot(self) -> DualPoseSnapshot | None:
        """Return an atomic A/B/object snapshot or None when tracking is stale."""

