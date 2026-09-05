"""Fail-closed production coordinator for the coupled dual-G1 policy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from omnicontact.runtime import BridgeState

from .dual_pose_provider import DualPoseProvider, DualPoseSnapshot
from .robot_session import RobotSession

if TYPE_CHECKING:
    from .scalebfm_residual_policy import DualScaleBFMResidualPolicy


class DeploymentState(str, Enum):
    ZERO_TORQUE = "zero_torque"
    LOCO_STANDING = "loco_standing"
    STOPPED = "stopped"
    WAITING = "waiting"
    DEFAULT_POSE = "default_pose"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAULT = "fault"


@dataclass(frozen=True)
class CoordinatorResult:
    ok: bool
    state: DeploymentState
    reason: str
    policy_step: Any | None = None
    processing_time_s: float = 0.0


class DualPolicyCoordinator:
    def __init__(
        self,
        robot_a: RobotSession,
        robot_b: RobotSession,
        pose_provider: DualPoseProvider,
        policy: DualScaleBFMResidualPolicy,
        *,
        enable_a: bool,
        enable_b: bool,
        state_timeout_s: float = 0.2,
        pose_timeout_s: float = 0.1,
        max_state_skew_s: float = 0.05,
        default_pose_ticks: int = 100,
        max_partner_position_error_m: float = 0.20,
        max_object_position_error_m: float = 0.20,
        max_box_size_error_m: float = 0.03,
        max_robot_orientation_error_rad: float = 0.35,
        max_object_orientation_error_rad: float = 0.35,
    ) -> None:
        self.robots = (robot_a, robot_b)
        self.pose_provider = pose_provider
        self.policy = policy
        self.enable = (int(enable_a), int(enable_b))
        self.state_timeout_s = float(state_timeout_s)
        self.pose_timeout_s = float(pose_timeout_s)
        self.max_state_skew_s = float(max_state_skew_s)
        self.default_pose_ticks = int(default_pose_ticks)
        if self.default_pose_ticks < 0:
            raise ValueError("default_pose_ticks must be non-negative")
        self.geometry_limits = (
            float(max_partner_position_error_m),
            float(max_object_position_error_m),
            float(max_box_size_error_m),
            float(max_robot_orientation_error_rad),
            float(max_object_orientation_error_rad),
        )
        self.state = DeploymentState.WAITING
        self.default_ticks_elapsed = 0
        self.last_geometry: dict[str, float] | None = None

    @staticmethod
    def _state_age_s(state: BridgeState) -> float:
        return max(0.0, (time.monotonic_ns() - state.packet_arrival_ns) * 1.0e-9)

    def _read_inputs(
        self,
    ) -> tuple[tuple[BridgeState, BridgeState] | None, DualPoseSnapshot | None, str]:
        state_a = self.robots[0].read(self.state_timeout_s)
        state_b = self.robots[1].read(self.state_timeout_s)
        if state_a is None or state_b is None:
            return None, self.pose_provider.get_snapshot(), "missing_bridge_state"
        if (
            self._state_age_s(state_a) > self.state_timeout_s
            or self._state_age_s(state_b) > self.state_timeout_s
        ):
            return None, self.pose_provider.get_snapshot(), "stale_bridge_state"
        skew = abs(state_a.packet_arrival_ns - state_b.packet_arrival_ns) * 1.0e-9
        if skew > self.max_state_skew_s:
            return None, self.pose_provider.get_snapshot(), "skewed_bridge_state"
        snapshot = self.pose_provider.get_snapshot()
        if snapshot is None:
            return (state_a, state_b), None, "missing_vive_snapshot"
        stamps = (
            snapshot.robot_a.stamp_s,
            snapshot.robot_b.stamp_s,
            snapshot.object.stamp_s,
        )
        age = time.monotonic() - min(stamps)
        if age < 0.0 or age > self.pose_timeout_s:
            return (state_a, state_b), None, "stale_vive_snapshot"
        return (state_a, state_b), snapshot, "ready"

    def _send_hold(self, states: tuple[BridgeState, BridgeState] | None) -> None:
        if states is None:
            states = tuple(robot.last_state for robot in self.robots)  # type: ignore[assignment]
        errors = []
        for robot, state in zip(self.robots, states):
            if state is not None:
                try:
                    robot.send_hold(state, enable=0)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    def step(self) -> CoordinatorResult:
        states, snapshot, reason = self._read_inputs()
        if states is None or snapshot is None:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            return CoordinatorResult(False, self.state, reason)
        try:
            if self.state == DeploymentState.WAITING:
                self.last_geometry = self.policy.initialize(
                    snapshot,
                    max_partner_position_error_m=self.geometry_limits[0],
                    max_object_position_error_m=self.geometry_limits[1],
                    max_box_size_error_m=self.geometry_limits[2],
                    max_robot_orientation_error_rad=self.geometry_limits[3],
                    max_object_orientation_error_rad=self.geometry_limits[4],
                )
                self.state = (
                    DeploymentState.DEFAULT_POSE
                    if self.default_pose_ticks > 0
                    else DeploymentState.EXECUTING
                )
                self.policy.reset_rollout()

            if self.state == DeploymentState.DEFAULT_POSE:
                for index, robot in enumerate(self.robots):
                    robot.send_target(
                        self.policy.start_joint_targets[index],
                        states[index],
                        enable=self.enable[index],
                    )
                self.default_ticks_elapsed += 1
                if self.default_ticks_elapsed >= self.default_pose_ticks:
                    self.policy.reset_rollout()
                    self.state = DeploymentState.EXECUTING
                return CoordinatorResult(True, self.state, "default_pose")

            if self.state == DeploymentState.EXECUTING:
                policy_step = self.policy.compute(states, snapshot)
                for index, robot in enumerate(self.robots):
                    robot.send_target(
                        policy_step.targets[index],
                        states[index],
                        enable=self.enable[index],
                    )
                if policy_step.complete:
                    self.state = DeploymentState.COMPLETE
                return CoordinatorResult(True, self.state, "policy_target", policy_step)

            self._send_hold(states)
            return CoordinatorResult(
                self.state == DeploymentState.COMPLETE, self.state, self.state.value
            )
        except BaseException:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            raise

    def close(self) -> None:
        self._send_hold(None)
        self.robots[0].close()
        self.robots[1].close()
