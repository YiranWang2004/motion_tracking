"""Policy-neutral bridge state and safety helpers for OmniContact."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from common.udp_transport import UDPRobotHigh
from omnicontact.contracts import (
    ObjectPose,
    PDCommand,
    ReferenceVisualization,
    RobotPose,
    TaskGoal,
)
from omnicontact.perception.object_pose import ExternalObjectPoseProvider
from omnicontact.visualization_udp import VisualizationSender


LOGGER = logging.getLogger(__name__)


class BridgePoseProvider(ExternalObjectPoseProvider):
    """Pose sink populated atomically from a simulation bridge state packet."""

    def __init__(self) -> None:
        super().__init__()
        import threading

        self._ready = threading.Event()
        self.error = None

    def start(self) -> dict:
        return {}

    def stop(self) -> None:
        self.clear()
        self._ready.clear()

    def wait_until_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(max(0.0, timeout_s))

    def publish_pair(self, robot_pose: RobotPose, object_pose: ObjectPose) -> None:
        super().publish_pair(robot_pose, object_pose)
        self._ready.set()


@dataclass(frozen=True)
class BridgeState:
    q_lab: np.ndarray
    dq_lab: np.ndarray
    quat_wxyz: np.ndarray
    gyro: np.ndarray
    buttons: dict[str, bool]
    state_receive_time_ns: int | None
    packet_seq: int
    packet_arrival_ns: int


def pose_pair_is_valid(
    robot_pose: RobotPose | None,
    object_pose: ObjectPose | None,
    *,
    max_age_s: float,
    min_confidence: float,
    now_s: float | None = None,
) -> bool:
    if robot_pose is None or object_pose is None:
        return False
    now = time.monotonic() if now_s is None else float(now_s)
    return (
        0.0 <= now - robot_pose.stamp_s <= max_age_s
        and 0.0 <= now - object_pose.stamp_s <= max_age_s
        and robot_pose.confidence >= min_confidence
        and object_pose.confidence >= min_confidence
    )


class CommandLimiter:
    """Validate, joint-limit and rate-limit absolute policy-order targets."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        max_target_delta: float,
    ) -> None:
        self.lower = np.asarray(lower, dtype=np.float32).reshape(29)
        self.upper = np.asarray(upper, dtype=np.float32).reshape(29)
        self.max_target_delta = float(max_target_delta)
        if self.max_target_delta <= 0.0:
            raise ValueError("max_target_delta must be positive")
        if np.any(self.lower >= self.upper):
            raise ValueError("invalid joint limits")
        self.last_target: np.ndarray | None = None

    def reset(self, measured_q: np.ndarray) -> None:
        measured = np.asarray(measured_q, dtype=np.float32).reshape(29)
        if not np.all(np.isfinite(measured)):
            raise ValueError("measured joint state is invalid")
        self.last_target = measured.copy()

    def apply(self, command: PDCommand, *, rate_limit: bool = True) -> PDCommand:
        target = np.clip(command.target_pos, self.lower, self.upper)
        if rate_limit and self.last_target is not None:
            target = np.clip(
                target,
                self.last_target - self.max_target_delta,
                self.last_target + self.max_target_delta,
            )
        self.last_target = target.astype(np.float32, copy=True)
        return PDCommand(target, command.kp, command.kd)

    def hold(self, measured_q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> PDCommand:
        measured = np.asarray(measured_q, dtype=np.float32).reshape(29)
        command = PDCommand(measured, kp, kd)
        return self.apply(command, rate_limit=False)


class MotionBridgeClient:
    """Thin checked client for the existing motion_tracking UDP bridge."""

    def __init__(
        self,
        udp_config: Any,
        *,
        pose_sink: BridgePoseProvider | None = None,
        visualization_sender: VisualizationSender | None = None,
        state_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.transport = UDPRobotHigh(udp_config)
        self.pose_sink = pose_sink
        self.visualization_sender = visualization_sender
        self.state_observer = state_observer
        self.last_seq: int | None = None
        self.skipped_packets = 0
        self._previous_buttons: dict[str, bool] | None = None
        self.button_rise: dict[str, bool] = {}
        self._carrybox_scene: dict[str, np.ndarray] | None = None
        self.command_count = 0
        self.last_command_gap_ms = 0.0
        self.max_command_gap_ms = 0.0
        self._last_command_attempt_ns: int | None = None
        self.latest_state: BridgeState | None = None
        self.sim_goal_position_w: np.ndarray | None = None

    def close(self) -> None:
        self.transport.close()
        visualization_sender = getattr(self, "visualization_sender", None)
        if visualization_sender is not None:
            visualization_sender.close()

    def read_next(self, timeout_s: float | None) -> BridgeState | None:
        packet = self.transport.read_next_state(
            after_seq=self.last_seq,
            timeout_s=timeout_s,
            with_meta=True,
        )
        if packet is None:
            return None
        if self.last_seq is not None and packet.seq > self.last_seq + 1:
            self.skipped_packets += int(packet.seq - self.last_seq - 1)
        self.last_seq = int(packet.seq)
        data = packet.data
        if not isinstance(data, dict):
            raise RuntimeError("bridge returned a non-mapping state")
        if self.state_observer is not None:
            self.state_observer(data)
        q = np.asarray(data["q"], dtype=np.float32).reshape(-1)
        dq = np.asarray(data["dq"], dtype=np.float32).reshape(-1)
        quat = np.asarray(data["quat_wxyz"], dtype=np.float32).reshape(-1)
        gyro = np.asarray(data["gyro"], dtype=np.float32).reshape(-1)
        if (
            q.shape != (29,)
            or dq.shape != (29,)
            or quat.shape != (4,)
            or gyro.shape != (3,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(dq))
            or not np.all(np.isfinite(quat))
            or not np.all(np.isfinite(gyro))
            or float(np.linalg.norm(quat)) < 1e-6
        ):
            raise RuntimeError("bridge returned an invalid G1 state")
        quat = quat / np.linalg.norm(quat)
        if self.pose_sink is not None:
            self._publish_sim_pose(data)
        raw_buttons = data.get("buttons", {})
        buttons = {
            name: bool(raw_buttons.get(name, False))
            for name in ("start", "stop", "A", "B", "up", "down")
        }
        if self._previous_buttons is None:
            self.button_rise = {name: False for name in buttons}
        else:
            self.button_rise = {
                name: (not self._previous_buttons[name]) and value
                for name, value in buttons.items()
            }
        self._previous_buttons = buttons
        state_time = data.get("state_receive_time_ns")
        state = BridgeState(
            q_lab=q.copy(),
            dq_lab=dq.copy(),
            quat_wxyz=quat.copy(),
            gyro=gyro.copy(),
            buttons=buttons,
            state_receive_time_ns=None if state_time is None else int(state_time),
            packet_seq=int(packet.seq),
            packet_arrival_ns=int(packet.recv_time_ns),
        )
        self.latest_state = state
        return state

    def _publish_sim_pose(self, data: dict[str, Any]) -> None:
        raw = data.get("sim_pose")
        if not isinstance(raw, dict):
            self.pose_sink.clear()
            self.sim_goal_position_w = None
            return
        robot = raw.get("robot")
        obj = raw.get("object")
        if not isinstance(robot, dict) or not isinstance(obj, dict):
            raise RuntimeError("simulation bridge returned an invalid sim_pose")
        stamp = time.monotonic()
        try:
            robot_pose = RobotPose(
                position_w=robot["position_w"],
                quaternion_xyzw=robot["quaternion_xyzw"],
                stamp_s=stamp,
                confidence=1.0,
            )
            object_pose = ObjectPose(
                position_w=obj["position_w"],
                quaternion_xyzw=obj["quaternion_xyzw"],
                half_extents=obj["half_extents"],
                stamp_s=stamp,
                confidence=1.0,
                linear_velocity_w=obj.get("linear_velocity_w"),
                angular_velocity_w=obj.get("angular_velocity_w"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"simulation bridge returned malformed task poses: {exc}") from exc
        raw_goal = raw.get("goal_position_w")
        if raw_goal is None:
            self.sim_goal_position_w = None
        else:
            try:
                self.sim_goal_position_w = TaskGoal(raw_goal).position_w
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"simulation bridge returned malformed task goal: {exc}"
                ) from exc
        self.pose_sink.publish_pair(robot_pose, object_pose)

    def set_carrybox_scene(self, object_pose: ObjectPose, goal: TaskGoal) -> None:
        """Configure sim-only start/goal planes using original runner semantics."""
        plane_z_offset = float(object_pose.half_extents[2]) + 0.01
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        start_position = object_pose.position_w.copy()
        start_position[2] -= plane_z_offset
        goal_position = goal.position_w.copy()
        goal_position[2] -= plane_z_offset
        self._carrybox_scene = {
            "start_plane_wxyz": np.concatenate((start_position, identity)),
            "goal_plane_wxyz": np.concatenate((goal_position, identity)),
        }

    @staticmethod
    def _reference_visualization_payload(
        visualization: ReferenceVisualization,
    ) -> dict[str, np.ndarray]:
        payload = {
            "left_wrist_wxyz": visualization.left_wrist_wxyz,
            "right_wrist_wxyz": visualization.right_wrist_wxyz,
            "torso_wxyz": visualization.torso_wxyz,
            "left_ankle_wxyz": visualization.left_ankle_wxyz,
            "right_ankle_wxyz": visualization.right_ankle_wxyz,
            "object_wxyz": visualization.object_wxyz,
            "contact": visualization.contact,
        }
        if visualization.ghost_base_wxyz is not None:
            payload["ghost_base_wxyz"] = visualization.ghost_base_wxyz
        if visualization.ghost_dof_pos is not None:
            payload["ghost_dof_pos"] = visualization.ghost_dof_pos
        return payload

    def publish_visualization(
        self,
        visualization: ReferenceVisualization | None = None,
    ) -> None:
        """Publish the read-only twin stream without sending a robot command."""
        visualization_sender = getattr(self, "visualization_sender", None)
        if visualization_sender is None or (
            self._carrybox_scene is None and visualization is None
        ):
            return
        reference_payload = (
            None
            if visualization is None
            else self._reference_visualization_payload(visualization)
        )
        visualization_sender.send(self._carrybox_scene, reference_payload)

    def send(
        self,
        command: PDCommand,
        *,
        enable: int,
        state: BridgeState,
        visualization: ReferenceVisualization | None = None,
    ) -> int:
        zeros = np.zeros(29, dtype=np.float32)
        extra_command = None
        if self.pose_sink is not None and (
            self._carrybox_scene is not None or visualization is not None
        ):
            omni_visualization: dict[str, Any] = {}
            if self._carrybox_scene is not None:
                omni_visualization["scene"] = self._carrybox_scene
            if visualization is not None:
                omni_visualization["reference"] = self._reference_visualization_payload(
                    visualization
                )
            extra_command = {"omnicontact_visualization": omni_visualization}
        self.publish_visualization(visualization)
        attempt_ns = time.monotonic_ns()
        self.last_command_gap_ms = getattr(self, "last_command_gap_ms", 0.0)
        self.max_command_gap_ms = getattr(self, "max_command_gap_ms", 0.0)
        last_attempt_ns = getattr(self, "_last_command_attempt_ns", None)
        if last_attempt_ns is not None:
            self.last_command_gap_ms = (
                attempt_ns - last_attempt_ns
            ) * 1e-6
            self.max_command_gap_ms = max(
                getattr(self, "max_command_gap_ms", 0.0),
                self.last_command_gap_ms,
            )
            if self.last_command_gap_ms >= 100.0:
                LOGGER.warning(
                    "CONTROL COMMAND GAP: %.3f ms since previous send attempt; "
                    "the default G1 bridge watchdog trips at 200 ms",
                    self.last_command_gap_ms,
                )
        self._last_command_attempt_ns = attempt_ns
        sequence = self.transport.send_command(
            q_des=command.target_pos,
            qd_des=zeros,
            kp=zeros if command.kp is None else command.kp,
            kd=zeros if command.kd is None else command.kd,
            enable=int(enable),
            extra_command=extra_command,
            state_receive_time_ns=state.state_receive_time_ns,
        )
        self.command_count = getattr(self, "command_count", 0) + 1
        return sequence

    def send_zero(self, state: BridgeState) -> int:
        zeros = np.zeros(29, dtype=np.float32)
        return self.send(PDCommand(zeros, zeros, zeros), enable=0, state=state)

    def send_damping(self, state: BridgeState, damping_kd: float = 8.0) -> int:
        zeros = np.zeros(29, dtype=np.float32)
        damping = np.full(29, float(damping_kd), dtype=np.float32)
        return self.send(PDCommand(zeros, zeros, damping), enable=0, state=state)
