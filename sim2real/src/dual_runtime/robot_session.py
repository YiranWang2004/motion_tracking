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
    max_target_delta: float = 0.02
    damping_kd: float = 8.0

    def __post_init__(self) -> None:
        if self.robot_id not in {"a", "b"}:
            raise ValueError("robot_id must be 'a' or 'b'")
        lower = np.asarray(self.lower, dtype=np.float32).reshape(29)
        upper = np.asarray(self.upper, dtype=np.float32).reshape(29)
        kp = np.asarray(self.kp, dtype=np.float32).reshape(29)
        kd = np.asarray(self.kd, dtype=np.float32).reshape(29)
        if np.any(lower >= upper):
            raise ValueError(f"invalid joint limits for robot {self.robot_id}")
        if np.any(kp < 0.0) or np.any(kd < 0.0):
            raise ValueError(f"PD gains must be non-negative for robot {self.robot_id}")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())
        object.__setattr__(self, "kp", kp.copy())
        object.__setattr__(self, "kd", kd.copy())


class RobotSession:
    """Per-robot bridge client, limiter, and last received state."""

    def __init__(self, config: RobotSessionConfig, client: MotionBridgeClient | None = None):
        self.config = config
        self.client = client or MotionBridgeClient(config.udp)
        self.limiter = CommandLimiter(
            config.lower, config.upper, config.max_target_delta
        )
        self.last_state: BridgeState | None = None

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
        self.client.send(command, enable=enable, state=state)
        return command

    def send_hold(self, state: BridgeState, *, enable: int = 0) -> PDCommand:
        kp = np.zeros(29, dtype=np.float32)
        kd = np.full(29, self.config.damping_kd, dtype=np.float32)
        command = self.limiter.hold(state.q_lab, kp, kd)
        self.client.send(command, enable=enable, state=state)
        return command

    def close(self) -> None:
        self.client.close()
