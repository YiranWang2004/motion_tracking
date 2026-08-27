"""Small, policy-neutral dual-G1 coordinator.

This module intentionally contains no ScaleBFM or OmniContact math.  It is the
hardware contract: one synchronized tick produces one PD command for A and
one for B.  A future joint policy can replace ``IndependentTargetPolicy``
without changing bridge routing or safety behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from omnicontact.runtime import BridgeState

from .robot_session import RobotSession


@dataclass
class IndependentTargetPolicy:
    """Deterministic low-amplitude target used for the first routing test."""

    joint_index: int
    offset_rad: float
    _baseline: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= int(self.joint_index) < 29:
            raise ValueError("joint_index must be in [0, 28]")
        if not np.isfinite(self.offset_rad):
            raise ValueError("offset_rad must be finite")

    def target(self, state: BridgeState) -> np.ndarray:
        if self._baseline is None:
            self._baseline = state.q_lab.copy()
        target = self._baseline.copy()
        target[int(self.joint_index)] += float(self.offset_rad)
        return target

    def reset(self) -> None:
        self._baseline = None


class IndependentDualCoordinator:
    """Read both bridges and send independent targets with fail-closed safety."""

    def __init__(
        self,
        robot_a: RobotSession,
        robot_b: RobotSession,
        policy_a: IndependentTargetPolicy,
        policy_b: IndependentTargetPolicy,
        *,
        state_timeout_s: float = 0.2,
        max_state_skew_s: float = 0.05,
        fail_closed: bool = True,
    ) -> None:
        self.a = robot_a
        self.b = robot_b
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.state_timeout_s = float(state_timeout_s)
        self.max_state_skew_s = float(max_state_skew_s)
        self.fail_closed = bool(fail_closed)
        if self.state_timeout_s <= 0.0 or self.max_state_skew_s < 0.0:
            raise ValueError("timeouts must be non-negative and state timeout positive")

    @staticmethod
    def _state_age_s(state: BridgeState) -> float:
        return max(0.0, (time.monotonic_ns() - state.packet_arrival_ns) * 1e-9)

    def step(self, *, enable_a: int = 0, enable_b: int = 0) -> tuple[bool, str]:
        state_a = self.a.read(self.state_timeout_s)
        state_b = self.b.read(self.state_timeout_s)
        valid = state_a is not None and state_b is not None
        if valid:
            assert state_a is not None and state_b is not None
            valid = (
                self._state_age_s(state_a) <= self.state_timeout_s
                and self._state_age_s(state_b) <= self.state_timeout_s
                and abs(state_a.packet_arrival_ns - state_b.packet_arrival_ns)
                * 1e-9
                <= self.max_state_skew_s
            )
        if not valid:
            reason = "missing_or_skewed_bridge_state"
            if self.fail_closed:
                if state_a is not None:
                    self.a.send_hold(state_a)
                if state_b is not None:
                    self.b.send_hold(state_b)
            return False, reason

        assert state_a is not None and state_b is not None
        if enable_a:
            self.a.send_target(self.policy_a.target(state_a), state_a, enable=1)
        else:
            self.a.send_hold(state_a, enable=0)
        if enable_b:
            self.b.send_target(self.policy_b.target(state_b), state_b, enable=1)
        else:
            self.b.send_hold(state_b, enable=0)
        return True, "independent_targets_sent"

    def close(self) -> None:
        for session in (self.a, self.b):
            state = getattr(session, "last_state", None)
            if state is not None:
                session.send_hold(state, enable=0)
        self.a.close()
        self.b.close()
