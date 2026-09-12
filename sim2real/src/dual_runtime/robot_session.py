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
            raise ValueError(
                f"torque limits must be positive for robot {self.robot_id}"
            )
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())
        object.__setattr__(self, "kp", kp.copy())
        object.__setattr__(self, "kd", kd.copy())
        object.__setattr__(
            self, "torque_limit", None if torque_limit is None else torque_limit.copy()
        )


class RobotSession:
    """Per-robot bridge client, limiter, and last received state."""

    def __init__(
        self, config: RobotSessionConfig, client: MotionBridgeClient | None = None
    ):
        self.config = config
        self.client = client or MotionBridgeClient(config.udp)
        self.limiter = CommandLimiter(
            config.lower, config.upper, config.max_target_delta
        )
        self.last_state: BridgeState | None = None
        self.last_command: PDCommand | None = None
        self.last_enable = 0
        self.last_command_diagnostics: dict[str, Any] = {}

    def read(self, timeout_s: float) -> BridgeState | None:
        state = self.client.read_next(timeout_s)
        if state is not None:
            self.last_state = state
        return state

    def prepare(self, state: BridgeState) -> None:
        self.limiter.reset(state.q_lab)

    def send_target(
        self, target: np.ndarray, state: BridgeState, *, enable: int
    ) -> PDCommand:
        return self.send_pd(
            PDCommand(
                np.asarray(target, dtype=np.float32), self.config.kp, self.config.kd
            ),
            state,
            enable=enable,
        )

    def send_pd(
        self,
        desired: PDCommand,
        state: BridgeState,
        *,
        enable: int,
        sim_control: dict[str, Any] | None = None,
        limit_estimated_torque: bool = True,
    ) -> PDCommand:
        if enable not in (0, 1):
            raise ValueError("enable must be 0 or 1")
        requested_target = desired.target_pos.copy()
        if not enable:
            # The hardware bridge forwards PD gains even when enable=0.
            # Encode disabled operation explicitly, including true zero torque.
            active = desired.kp is not None and np.any(desired.kp != 0)
            kd = np.full(29, self.config.damping_kd) if active else desired.kd
            desired = PDCommand(state.q_lab.copy(), np.zeros(29), kd)
        if self.limiter.last_target is None:
            self.prepare(state)
        command = self.limiter.apply(desired)
        diagnostics = {
            "command_requested_target": requested_target,
            "command_pre_limit_target": desired.target_pos.copy(),
            "command_post_target_limit": command.target_pos.copy(),
            "torque_limit_enabled": bool(limit_estimated_torque and self.config.torque_limit is not None),
            "torque_limit_triggered": np.zeros(29, dtype=bool),
            "estimated_torque_pre_limit": np.full(29, np.nan, dtype=np.float32),
            "estimated_torque_clipped": np.full(29, np.nan, dtype=np.float32),
        }
        kp = np.zeros(29) if command.kp is None else command.kp
        kd = np.zeros(29) if command.kd is None else command.kd
        if limit_estimated_torque and self.config.torque_limit is not None:
            estimated = kp * (command.target_pos - state.q_lab) - kd * state.dq_lab
            limited = np.clip(
                estimated, -self.config.torque_limit, self.config.torque_limit
            )
            diagnostics["estimated_torque_pre_limit"] = estimated.copy()
            diagnostics["estimated_torque_clipped"] = limited.copy()
            diagnostics["torque_limit_triggered"] = np.abs(estimated) > self.config.torque_limit
            safe_target = command.target_pos.copy()
            active = kp > 1.0e-6
            safe_target[active] = (
                state.q_lab[active]
                + (limited[active] + kd[active] * state.dq_lab[active]) / kp[active]
            )
            safe_target = np.clip(
                safe_target, self.config.lower, self.config.upper
            ).astype(np.float32)
            command = PDCommand(safe_target, kp, kd)
            self.limiter.last_target = safe_target.copy()
        kwargs = {} if sim_control is None else {"sim_control": sim_control}
        self.client.send(command, enable=enable, state=state, **kwargs)
        self.last_enable = enable
        self.last_command = command
        self.last_command_diagnostics = diagnostics
        return command

    def send_hold(self, state: BridgeState, *, enable: int = 0) -> PDCommand:
        kp = np.zeros(29, dtype=np.float32)
        kd = np.full(29, self.config.damping_kd, dtype=np.float32)
        command = self.limiter.hold(state.q_lab, kp, kd)
        self.client.send(command, enable=enable, state=state)
        self.last_enable = enable
        self.last_command = command
        # A fault/stop hold must not inherit diagnostics from the previous policy command.
        self.last_command_diagnostics = {}
        return command

    def close(self) -> None:
        self.client.close()
