"""Object-pose source interface; Vive is an implementation, not a task dependency."""

import threading
import time
from abc import ABC, abstractmethod

from omnicontact.contracts import ObjectPose, RobotPose


class ObjectPoseProvider(ABC):
    @abstractmethod
    def get_pose(self) -> ObjectPose | None:
        """Return the latest world-frame pose, or None when no valid measurement exists."""

    def get_robot_pose(self) -> RobotPose | None:
        """Return a world-frame pelvis pose when this source provides one."""
        return None

    def get_poses(self) -> tuple[RobotPose | None, ObjectPose | None]:
        """Return a robot/object snapshot.

        Providers backed by one sensor packet should override this method so
        callers never combine states from two different packets.
        """
        return self.get_robot_pose(), self.get_pose()


class SimObjectPoseProvider(ObjectPoseProvider):
    """Adapter for named free bodies exposed by the carry-box MuJoCo environment."""

    def __init__(self, env, body_name: str = "box"):
        self.env = env
        self.body_name = body_name

    def get_pose(self) -> ObjectPose | None:
        state = self.env.get_object_state(self.body_name)
        if state is None:
            return None
        return ObjectPose(
            position_w=state["position_w"],
            quaternion_xyzw=state["quaternion_xyzw"],
            half_extents=state["half_extents"],
            linear_velocity_w=state.get("linear_velocity_w"),
            angular_velocity_w=state.get("angular_velocity_w"),
            confidence=1.0,
            stamp_s=time.monotonic(),
        )

    def get_robot_pose(self) -> RobotPose | None:
        return RobotPose(
            position_w=self.env.base_pos,
            quaternion_xyzw=self.env.base_quat,
            stamp_s=time.monotonic(),
        )

    def get_poses(self) -> tuple[RobotPose | None, ObjectPose | None]:
        # Both states originate from the same MuJoCo data snapshot.
        return self.get_robot_pose(), self.get_pose()


class ExternalObjectPoseProvider(ObjectPoseProvider):
    """Thread-safe sink for an external tracker adapter, including Vive.

    The Vive driver belongs outside RoboJuDo because its transport differs by
    installation. That driver converts its rigid-body pose into the calibrated
    RoboJuDo odometry/world frame and calls ``publish`` at its native rate.
    """

    def __init__(self):
        self._pose: ObjectPose | None = None
        self._robot_pose: RobotPose | None = None
        self._lock = threading.Lock()

    def publish(self, pose: ObjectPose) -> None:
        self.publish_object_pose(pose)

    def publish_object_pose(self, pose: ObjectPose) -> None:
        with self._lock:
            self._pose = pose

    def publish_robot_pose(self, pose: RobotPose) -> None:
        with self._lock:
            self._robot_pose = pose

    def publish_pair(self, robot_pose: RobotPose, object_pose: ObjectPose) -> None:
        """Atomically replace both poses from one tracker/network sample."""
        with self._lock:
            self._robot_pose = robot_pose
            self._pose = object_pose

    def clear(self) -> None:
        """Immediately invalidate both poses, for example after tracking loss."""
        with self._lock:
            self._robot_pose = None
            self._pose = None

    def get_pose(self) -> ObjectPose | None:
        with self._lock:
            return self._pose

    def get_robot_pose(self) -> RobotPose | None:
        with self._lock:
            return self._robot_pose

    def get_poses(self) -> tuple[RobotPose | None, ObjectPose | None]:
        with self._lock:
            return self._robot_pose, self._pose

