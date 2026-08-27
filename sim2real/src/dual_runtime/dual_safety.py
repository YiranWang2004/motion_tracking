"""Safety predicates shared by dual policy implementations."""

from __future__ import annotations

import time

from omnicontact.runtime import BridgeState


def states_are_fresh_and_synchronized(
    state_a: BridgeState | None,
    state_b: BridgeState | None,
    *,
    timeout_s: float,
    max_skew_s: float,
) -> bool:
    if state_a is None or state_b is None:
        return False
    now_ns = time.monotonic_ns()
    if (now_ns - state_a.packet_arrival_ns) * 1e-9 > timeout_s:
        return False
    if (now_ns - state_b.packet_arrival_ns) * 1e-9 > timeout_s:
        return False
    return abs(state_a.packet_arrival_ns - state_b.packet_arrival_ns) * 1e-9 <= max_skew_s

