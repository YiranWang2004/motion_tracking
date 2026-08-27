import numpy as np

from dual_runtime.coordinator import IndependentDualCoordinator, IndependentTargetPolicy
from omnicontact.contracts import PDCommand
from omnicontact.runtime import BridgeState


def _state(seq: int, arrival_ns: int) -> BridgeState:
    return BridgeState(
        q_lab=np.zeros(29, dtype=np.float32),
        dq_lab=np.zeros(29, dtype=np.float32),
        quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        buttons={},
        state_receive_time_ns=None,
        packet_seq=seq,
        packet_arrival_ns=arrival_ns,
    )


class FakeSession:
    def __init__(self, state):
        self.state = state
        self.sent = []

    def read(self, timeout_s):
        return self.state

    def send_target(self, target, state, *, enable):
        self.sent.append((target.copy(), enable))
        return PDCommand(target)

    def send_hold(self, state, *, enable=0):
        self.sent.append((state.q_lab.copy(), enable))

    def close(self):
        pass


def test_independent_targets_are_distinct_and_routed_to_each_session():
    now = __import__("time").monotonic_ns()
    a = FakeSession(_state(1, now))
    b = FakeSession(_state(1, now))
    coordinator = IndependentDualCoordinator(
        a,
        b,
        IndependentTargetPolicy(0, 0.03),
        IndependentTargetPolicy(0, -0.03),
        max_state_skew_s=0.01,
    )
    ok, reason = coordinator.step(enable_a=1, enable_b=1)
    assert ok and reason == "independent_targets_sent"
    np.testing.assert_allclose(a.sent[0][0][0], 0.03)
    np.testing.assert_allclose(b.sent[0][0][0], -0.03)


def test_target_is_relative_to_first_state_not_a_moving_measurement():
    policy = IndependentTargetPolicy(21, 0.03)
    first = _state(1, __import__("time").monotonic_ns())
    first.q_lab[21] = 0.5
    second = _state(2, __import__("time").monotonic_ns())
    second.q_lab[21] = 0.8
    np.testing.assert_allclose(policy.target(first)[21], 0.53)
    np.testing.assert_allclose(policy.target(second)[21], 0.53)


def test_only_selected_robot_receives_enabled_target():
    now = __import__("time").monotonic_ns()
    a = FakeSession(_state(1, now))
    b = FakeSession(_state(1, now))
    coordinator = IndependentDualCoordinator(
        a,
        b,
        IndependentTargetPolicy(21, 0.03),
        IndependentTargetPolicy(21, -0.03),
    )
    ok, _ = coordinator.step(enable_a=1, enable_b=0)
    assert ok
    assert a.sent[0][1] == 1
    assert b.sent[0][1] == 0


def test_skewed_state_fails_closed_for_both_sessions():
    now = __import__("time").monotonic_ns()
    a = FakeSession(_state(1, now))
    b = FakeSession(_state(1, now + 100_000_000))
    coordinator = IndependentDualCoordinator(
        a,
        b,
        IndependentTargetPolicy(0, 0.03),
        IndependentTargetPolicy(0, -0.03),
        max_state_skew_s=0.001,
    )
    ok, reason = coordinator.step(enable_a=1, enable_b=1)
    assert not ok and reason == "missing_or_skewed_bridge_state"
    assert len(a.sent) == 1 and len(b.sent) == 1
