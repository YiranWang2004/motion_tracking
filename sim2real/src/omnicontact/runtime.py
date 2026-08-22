"""Policy-neutral bridge state and safety helpers for OmniContact."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.udp_transport import UDPRobotHigh
from omnicontact.contracts import ObjectPose, PDCommand, RobotPose


@dataclass(frozen=True)
class BridgeState:
    q_lab: np.ndarray
    dq_lab: np.ndarray
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

    def __init__(self, udp_config: Any) -> None:
        self.transport = UDPRobotHigh(udp_config)
        self.last_seq: int | None = None
        self.skipped_packets = 0
        self._previous_buttons: dict[str, bool] | None = None
        self.button_rise: dict[str, bool] = {}

    def close(self) -> None:
        self.transport.close()

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
        q = np.asarray(data["q"], dtype=np.float32).reshape(-1)
        dq = np.asarray(data["dq"], dtype=np.float32).reshape(-1)
        gyro = np.asarray(data["gyro"], dtype=np.float32).reshape(-1)
        if (
            q.shape != (29,)
            or dq.shape != (29,)
            or gyro.shape != (3,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(dq))
            or not np.all(np.isfinite(gyro))
        ):
            raise RuntimeError("bridge returned an invalid G1 state")
        raw_buttons = data.get("buttons", {})
        buttons = {
            name: bool(raw_buttons.get(name, False))
            for name in ("start", "stop", "A", "up", "down")
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
        return BridgeState(
            q_lab=q.copy(),
            dq_lab=dq.copy(),
            gyro=gyro.copy(),
            buttons=buttons,
            state_receive_time_ns=None if state_time is None else int(state_time),
            packet_seq=int(packet.seq),
            packet_arrival_ns=int(packet.recv_time_ns),
        )

    def send(self, command: PDCommand, *, enable: int, state: BridgeState) -> int:
        zeros = np.zeros(29, dtype=np.float32)
        return self.transport.send_command(
            q_des=command.target_pos,
            qd_des=zeros,
            kp=zeros if command.kp is None else command.kp,
            kd=zeros if command.kd is None else command.kd,
            enable=int(enable),
            state_receive_time_ns=state.state_receive_time_ns,
        )

    def send_zero(self, state: BridgeState) -> int:
        zeros = np.zeros(29, dtype=np.float32)
        return self.send(PDCommand(zeros, zeros, zeros), enable=0, state=state)

    def send_damping(self, state: BridgeState, damping_kd: float = 8.0) -> int:
        zeros = np.zeros(29, dtype=np.float32)
        damping = np.full(29, float(damping_kd), dtype=np.float32)
        return self.send(PDCommand(zeros, zeros, damping), enable=0, state=state)
