"""The same operator sequence and PD semantics must work on either backend."""
import numpy as np
import pytest
from test_dual_sim_control import runtime, tick
from dual_runtime.policy_coordinator import DeploymentState as S
from dual_runtime.runtime_config import shared_control_settings
from dual_runtime.deployment_recorder import DualDeploymentRecorder
from omnicontact.contracts import PDCommand

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_shared_gates_transition_completion_and_stop(runtime, backend):
    coordinator, clients, policy, _ = runtime
    coordinator.input_mode = backend
    coordinator.transition_ticks = 4
    coordinator._buttons_initialized = False
    coordinator.phase_target_delta = shared_control_settings({})["phase_target_delta"]
    assert tick(runtime, "start").state == S.ZERO_TORQUE  # held at startup
    tick(runtime)
    assert tick(runtime, "start").state == S.DEFAULT_POSE
    np.testing.assert_allclose(clients[0].sent[-1][0].target_pos, .02)
    assert tick(runtime, "B").state == S.DEFAULT_POSE  # early B ignored
    tick(runtime)
    tick(runtime)
    assert tick(runtime, "B").state == S.LOCO_STANDING
    tick(runtime)
    assert tick(runtime, "A").state == S.EXECUTING
    assert coordinator.robots[0].limiter.max_target_delta == 1.0
    policy.complete = True
    assert tick(runtime).state == S.LOCO_STANDING
    assert tick(runtime, "stop", robot=1).state == S.STOPPED
    for client in clients:
        command, options = client.sent[-1]
        assert options["enable"] == 0
        np.testing.assert_array_equal(command.kp, 0)
        np.testing.assert_array_equal(command.kd, 8)
        assert ("sim_control" in options) == (backend == "sim")

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_disabled_has_no_position_gain_and_zero_is_zero(runtime, backend):
    coordinator, clients, _, _ = runtime
    coordinator.input_mode = backend
    coordinator.enable = (0, 0)
    tick(runtime)
    for c in clients:
        np.testing.assert_array_equal(c.sent[-1][0].kp, 0)
        np.testing.assert_array_equal(c.sent[-1][0].kd, 0)
    tick(runtime, "start")
    for c in clients:
        np.testing.assert_array_equal(c.sent[-1][0].kp, 0)
        np.testing.assert_array_equal(c.sent[-1][0].kd, 8)

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_pose_loss_records_failure_and_damps_both(runtime, backend, tmp_path):
    coordinator, clients, _, _ = runtime
    coordinator.input_mode = backend
    recorder = DualDeploymentRecorder(tmp_path, {})
    recorder.record(coordinator, tick(runtime))
    coordinator.pose_provider.get_snapshot = lambda: None
    result = tick(runtime)
    assert not result.ok and result.state == S.FAULT
    recorder.record(coordinator, result)
    log = np.load(recorder.save())
    assert not log["snapshot_valid"][-1]
    assert not log["ok"][-1]
    assert "missing" in log["reason"][-1]
    np.testing.assert_array_equal(log["command_kp"][-1], 0)
    np.testing.assert_array_equal(log["command_kd"][-1], 8)

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_shared_object_guard_stops_before_policy_send(runtime, backend):
    coordinator, clients, _, _ = runtime
    coordinator.input_mode = backend
    coordinator.task_safety = shared_control_settings({})["task_safety"]
    tick(runtime)
    tick(runtime, "start")
    tick(runtime, "B")
    with pytest.raises(RuntimeError, match="object.position_z"):
        tick(runtime, "A")
    assert coordinator.state == S.FAULT
    for c in clients:
        assert c.sent[-1][1]["enable"] == 0
        np.testing.assert_array_equal(c.sent[-1][0].kp, 0)

def test_hardware_accepts_independent_bridge_timestamps(runtime):
    coordinator, clients, _, _ = runtime
    coordinator.input_mode = "hardware"
    clients[0].stamp, clients[1].stamp = 100, 200
    assert tick(runtime).ok
    assert tick(runtime, "start").state == S.DEFAULT_POSE
    assert tick(runtime, "B").state == S.LOCO_STANDING
    # Repeated device stamps do not trigger the simulator retry cache.
    assert tick(runtime).reason == "loco_mode_standing"


def test_hold_attempts_both_bridges_when_one_send_fails(runtime):
    coordinator, clients, _, _ = runtime
    tick(runtime)
    def fail(*args, **kwargs):
        raise OSError("robot A socket failed")
    clients[0].send = fail
    with pytest.raises(OSError, match="robot A"):
        coordinator._send_hold(None)
    assert clients[1].sent[-1][1]["enable"] == 0
    np.testing.assert_array_equal(clients[1].sent[-1][0].kp, 0)

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_motion_lean_is_not_an_upright_standing_fault(runtime, backend):
    from dataclasses import replace
    coordinator, clients, _, _ = runtime
    coordinator.input_mode = backend
    coordinator.max_tilt_rad = .7
    tick(runtime, "start")
    tick(runtime, "B")
    tick(runtime, "A")
    for client in clients:
        read = client.read_next
        client.read_next = lambda timeout, read=read: replace(
            read(timeout), quat_wxyz=np.array([np.cos(.4), 0, np.sin(.4), 0]))
    assert tick(runtime).state == S.EXECUTING
    assert all(c.sent[-1][1]["enable"] == 1 for c in clients)
    coordinator.state = S.LOCO_STANDING
    with pytest.raises(RuntimeError, match="tilt"):
        tick(runtime)

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_object_guard_checks_last_applied_reference_not_next_target(runtime, backend):
    from types import SimpleNamespace
    from test_dual_scalebfm_deploy import fresh_snapshot
    coordinator, clients, policy, _ = runtime
    coordinator.input_mode = backend
    coordinator.task_safety = shared_control_settings({})["task_safety"]
    target = fresh_snapshot().object.position_w.copy()
    policy.reference.frame = lambda frame: (SimpleNamespace(object_pos_w=target.copy()), None)
    tick(runtime, "start")
    tick(runtime, "B")
    assert tick(runtime, "A").state == S.EXECUTING
    target[2] += .2
    # This snapshot is the response to the preceding command, whose target matched.
    assert tick(runtime).state == S.EXECUTING
    with pytest.raises(RuntimeError, match="object.position_z"):
        tick(runtime)  # The new command has now been applied, but object did not follow.

@pytest.mark.parametrize("backend", ["sim", "hardware"])
def test_task_entry_preserves_dynamic_policy_targets(runtime, backend):
    coordinator, clients, _, _ = runtime
    coordinator.input_mode = backend
    coordinator.phase_target_delta = shared_control_settings({})["phase_target_delta"]
    tick(runtime, "start")
    tick(runtime, "B")
    result = tick(runtime, "A")
    for client, desired in zip(clients, result.policy_step.targets):
        np.testing.assert_allclose(client.sent[-1][0].target_pos, desired)

def test_a_preserves_previous_command_as_limiter_anchor(runtime):
    coordinator, clients, _, _ = runtime
    coordinator.phase_target_delta = {"executing": .02}
    tick(runtime, "start")
    tick(runtime, "B")
    previous = [client.sent[-1][0].target_pos.copy() for client in clients]
    result = tick(runtime, "A")
    for client, target, anchor in zip(clients, result.policy_step.targets, previous):
        np.testing.assert_allclose(client.sent[-1][0].target_pos,
                                   np.clip(target, anchor - .02, anchor + .02))
