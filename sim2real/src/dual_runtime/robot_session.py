"""One G1 UDP bridge session used by the dual coordinator.

The session owns only per-robot state.  Keeping this object independent makes
it impossible for the coordinator to accidentally reuse robot A's sequence,
button edge, limiter, or UDP socket for robot B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from omnicontact.contracts import PDCommand
from omnicontact.runtime import BridgeState, CommandLimiter, MotionBridgeClient


@dataclass(frozen=True)
class RobotSessionConfig:
    robot_id: str
    udp: dict[str, Any]
    lower: np.ndarray
    upper: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray | None = None
    max_target_delta: float = 0.02
    damping_kd: float = 8.0

    def __post_init__(self) -> None:
        if self.robot_id not in {"a", "b"}:
            raise ValueError("robot_id must be 'a' or 'b'")
        lower = np.asarray(self.lower, dtype=np.float32).reshape(29)
        upper = np.asarray(self.upper, dtype=np.float32).reshape(29)
        kp = np.asarray(self.kp, dtype=np.float32).reshape(29)
        kd = np.asarray(self.kd, dtype=np.float32).reshape(29)
        torque_limit = (
            None
            if self.torque_limit is None
            else np.asarray(self.torque_limit, dtype=np.float32).reshape(29)
        )
        if np.any(lower >= upper):
            raise ValueError(f"invalid joint limits for robot {self.robot_id}")
        if np.any(kp < 0.0) or np.any(kd < 0.0):
            raise ValueError(f"PD gains must be non-negative for robot {self.robot_id}")
        if torque_limit is not None and (
            np.any(torque_limit <= 0.0) or not np.all(np.isfinite(torque_limit))
        ):
            raise ValueError(f"torque limits must be positive for robot {self.robot_id}")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())
        object.__setattr__(self, "kp", kp.copy())
        object.__setattr__(self, "kd", kd.copy())
        object.__setattr__(
            self, "torque_limit", None if torque_limit is None else torque_limit.copy()
        )


class RobotSession:
    """Per-robot bridge client, limiter, and last received state."""

    def __init__(self, config: RobotSessionConfig, client: MotionBridgeClient | None = None):
        self.config = config
        self.client = client or MotionBridgeClient(config.udp)
        self.limiter = CommandLimiter(
            config.lower, config.upper, config.max_target_delta
        )
        self.last_state: BridgeState | None = None
        self.last_command: PDCommand | None = None

    def read(self, timeout_s: float) -> BridgeState | None:
        state = self.client.read_next(timeout_s)
        if state is not None:
            self.last_state = state
        return state

    def prepare(self, state: BridgeState) -> None:
        self.limiter.reset(state.q_lab)

    def send_target(self, target: np.ndarray, state: BridgeState, *, enable: int) -> PDCommand:
        if self.limiter.last_target is None:
            self.prepare(state)
        command = self.limiter.apply(
            PDCommand(np.asarray(target, dtype=np.float32), self.config.kp, self.config.kd)
        )
        if self.config.torque_limit is not None:
            estimated = (
                self.config.kp * (command.target_pos - state.q_lab)
                - self.config.kd * state.dq_lab
            )
            limited = np.clip(
                estimated, -self.config.torque_limit, self.config.torque_limit
            )
            safe_target = command.target_pos.copy()
            active = self.config.kp > 1.0e-6
            safe_target[active] = (
                state.q_lab[active]
                + (limited[active] + self.config.kd[active] * state.dq_lab[active])
                / self.config.kp[active]
            )
            safe_target = np.clip(
                safe_target, self.config.lower, self.config.upper
            ).astype(np.float32)
            command = PDCommand(safe_target, self.config.kp, self.config.kd)
            self.limiter.last_target = safe_target.copy()
        self.client.send(command, enable=enable, state=state)
        self.last_command = command
        return command

    def send_hold(self, state: BridgeState, *, enable: int = 0) -> PDCommand:
        kp = np.zeros(29, dtype=np.float32)
        kd = np.full(29, self.config.damping_kd, dtype=np.float32)
        command = self.limiter.hold(state.q_lab, kp, kd)
        self.client.send(command, enable=enable, state=state)
        self.last_command = command
        return command

    def close(self) -> None:
        self.client.close()
