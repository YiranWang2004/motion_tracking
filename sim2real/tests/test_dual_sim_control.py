from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from dual_runtime.policy_coordinator import DeploymentState
from dual_runtime.robot_session import RobotSession, RobotSessionConfig
from dual_runtime.sim_control import InteractiveDualCoordinator, load_default_command
from omnicontact.contracts import PDCommand
from test_dual_scalebfm_deploy import state, fresh_snapshot, FakeCoupledPolicy


class Client:
    def __init__(self):
        self.buttons = {}
        self.sent = []
        self.stamp = None

    def read_next(self, timeout):
        value = state()
        value.buttons.update(self.buttons)
        from dataclasses import replace

        return replace(value, state_receive_time_ns=self.stamp)

    def send(self, command, **kwargs):
        self.sent.append((command, kwargs))

    def close(self):
        pass


class Loco:
    def __init__(self, target):
        self.target = target
        self.resets = 0
        self.calls = 0

    def reset(self):
        self.resets += 1

    def compute(self, current):
        self.calls += 1
        return PDCommand(np.full(29, self.target), np.full(29, 12), np.full(29, 3))


class Policy(FakeCoupledPolicy):
    def __init__(self):
        super().__init__()
        self.kp = np.full(29, 20)
        self.kd = np.full(29, 2)
        self.calls = 0
        self.complete = False
        self.reference = SimpleNamespace(
            frame=lambda frame: (SimpleNamespace(object_pos_w=np.zeros(3)), None)
        )

    def compute(self, states, snapshot):
        self.calls += 1
        step = super().compute(states, snapshot)
        step.complete = self.complete
        return step


@pytest.fixture
def runtime():
    clients = [Client(), Client()]
    sessions = [
        RobotSession(
            RobotSessionConfig(
                robot_id=name,
                udp={},
                lower=np.full(29, -3),
                upper=np.full(29, 3),
                kp=np.ones(29),
                kd=np.ones(29),
                max_target_delta=1.0,
            ),
            client=client,
        )
        for name, client in zip(("a", "b"), clients)
    ]
    policy = Policy()
    locos = (Loco(0.3), Loco(-0.3))
    default = PDCommand(np.full(29, 0.1), np.full(29, 100), np.full(29, 4))
    coordinator = InteractiveDualCoordinator(
        *sessions,
        SimpleNamespace(get_snapshot=fresh_snapshot),
        policy,
        enable_a=True,
        enable_b=True,
        loco_modes=locos,
        default_command=default,
    )
    return coordinator, clients, policy, locos


def tick(runtime, button=None, robot=0):
    coordinator, clients, _, _ = runtime
    for client in clients:
        client.buttons = {}
    if button:
        clients[robot].buttons[button] = True
    return coordinator.step()


def test_manual_gates_loco_gains_and_return_to_standing(runtime):
    coordinator, clients, policy, locos = runtime
    for _ in range(4):
        assert tick(runtime).state == DeploymentState.ZERO_TORQUE
    assert not policy.initialized and policy.calls == 0
    for client in clients:
        np.testing.assert_array_equal(client.sent[-1][0].kp, 0)
        assert client.sent[-1][1]["sim_control"]["phase"] == "zero_torque"
    assert tick(runtime, "start").state == DeploymentState.DEFAULT_POSE
    for _ in range(4):
        assert tick(runtime).state == DeploymentState.DEFAULT_POSE
    for client in clients:
        np.testing.assert_array_equal(client.sent[-1][0].kp, 100)
    assert tick(runtime, "B").state == DeploymentState.LOCO_STANDING
    assert not policy.initialized
    assert locos[0].resets == locos[1].resets == 1
    np.testing.assert_allclose(clients[0].sent[-1][0].target_pos, 0.3)
    np.testing.assert_allclose(clients[1].sent[-1][0].target_pos, -0.3)
    np.testing.assert_array_equal(clients[1].sent[-1][0].kp, 12)
    for _ in range(3):
        tick(runtime)
    assert policy.calls == 0
    assert tick(runtime, "A").state == DeploymentState.EXECUTING
    assert policy.initialized and policy.calls == 1
    controls = [client.sent[-1][1]["sim_control"] for client in clients]
    assert (
        controls[0]
        == controls[1]
        == {"phase": "executing", "frame": 1, "reference_position_w": [0.0, 0.0, 0.0]}
    )
    policy.complete = True
    assert tick(runtime).state == DeploymentState.LOCO_STANDING
    tick(runtime)
    assert clients[0].sent[-1][1]["sim_control"]["phase"] == "loco_standing"
    assert locos[0].resets == locos[1].resets == 2
    tick(runtime, "A")
    assert policy.calls == 2  # Finished references are not silently restarted.
    assert tick(runtime, "stop", robot=1).state == DeploymentState.STOPPED
    assert all(client.sent[-1][1]["enable"] == 0 for client in clients)


@pytest.mark.parametrize("button", ["A", "B"])
def test_early_buttons_cannot_skip_start(runtime, button):
    assert tick(runtime, button).state == DeploymentState.ZERO_TORQUE
    assert not runtime[2].initialized
    tick(runtime)
    tick(runtime, "start")
    tick(runtime)
    assert tick(runtime, "A").state == DeploymentState.DEFAULT_POSE


@pytest.mark.parametrize(
    "phase_keys", [[], ["start"], ["start", "B"], ["start", "B", "A"]]
)
def test_stop_from_every_phase_disables_both(runtime, phase_keys):
    for key in phase_keys:
        tick(runtime, key)
        tick(runtime)
    assert tick(runtime, "stop").state == DeploymentState.STOPPED
    for client in runtime[1]:
        command, kwargs = client.sent[-1]
        assert kwargs["enable"] == 0
        assert kwargs["sim_control"]["phase"] == "stopped"
        np.testing.assert_array_equal(command.kp, 0)


def test_preflight_error_disables_both(runtime):
    def fail(*args, **kwargs):
        raise RuntimeError("geometry mismatch")

    runtime[2].initialize = fail
    for key in ("start", "B"):
        tick(runtime, key)
        tick(runtime)
    with pytest.raises(RuntimeError, match="geometry mismatch"):
        tick(runtime, "A")
    assert runtime[0].state == DeploymentState.FAULT
    assert all(client.sent[-1][1]["enable"] == 0 for client in runtime[1])


def test_real_default_and_loco_assets_use_same_joint_order():
    from omnicontact.loco_mode import LocoModePolicy
    from dual_runtime.constants import POLICY_JOINT_NAMES

    assets = Path(__file__).resolve().parents[1] / "config/g1/omnicontact"
    default = load_default_command(assets)
    assert default.target_pos.shape == (29,)
    assert default.kp[2] == 300  # waist yaw after MuJoCo -> policy reorder
    a, b = (LocoModePolicy(assets, list(POLICY_JOINT_NAMES)) for _ in range(2))
    before = b.hidden_state.copy()
    result = a.compute(state())
    assert np.all(np.isfinite(result.target_pos))
    np.testing.assert_array_equal(b.hidden_state, before)
    assert not np.shares_memory(a.hidden_state, b.hidden_state)


def test_phase_metadata_survives_real_command_payload():
    from omnicontact.runtime import MotionBridgeClient
    from common.udp_transport import _command_payload
    from dual_scalebfm_sim2sim import DualScaleBFMSim2Sim

    packets = []

    def capture(**kwargs):
        packets.append(_command_payload(**kwargs))
        return 1

    client = MotionBridgeClient.__new__(MotionBridgeClient)
    client.pose_sink = None
    client.transport = SimpleNamespace(send_command=capture)
    current = state()
    # BridgeState is frozen; create a timestamped replacement for the handshake.
    from dataclasses import replace

    current = replace(current, state_receive_time_ns=123)
    control = {
        "phase": "executing",
        "frame": 42,
        "reference_position_w": [1.0, 2.0, 3.0],
    }
    client.send(
        PDCommand(np.zeros(29), np.ones(29), np.ones(29)),
        enable=1,
        state=current,
        sim_control=control,
    )
    parsed = DualScaleBFMSim2Sim.parse_command(packets[-1])
    assert parsed.phase == "executing" and parsed.frame == 42
    assert parsed.state_time_ns == 123 and parsed.reference_position == (1.0, 2.0, 3.0)
    # Existing single-robot/hardware sends do not acquire simulation metadata.
    client.send(PDCommand(np.zeros(29)), enable=0, state=current)
    assert "extra_command" not in packets[-1]
    assert DualScaleBFMSim2Sim.parse_command(packets[-1]).phase == "stopped"


def test_mujoco_default_to_two_real_loco_modes(tmp_path, monkeypatch):
    """Run the real standing networks/PD/physics, using only a synthetic task bundle."""
    from dataclasses import replace
    import test_dual_scalebfm_sim2sim as sim_tests
    from omnicontact.loco_mode import LocoModePolicy
    from dual_runtime.constants import POLICY_JOINT_NAMES
    from dual_runtime.sim_pose_provider import DualSimulationPoseProvider
    from common.udp_transport import _command_payload
    from omnicontact.runtime import MotionBridgeClient

    # Reuse the explicitly synthetic scene fixture; no actor checkpoint needed.
    sim_tests.configure_test_scene(tmp_path, monkeypatch)
    sim = sim_tests.make_sim()
    assets = Path(__file__).resolve().parents[1] / "config/g1/omnicontact"
    provider = DualSimulationPoseProvider()
    commands = [None, None]
    clients = []
    for index, binding in enumerate(sim.bindings):
        client = MotionBridgeClient.__new__(MotionBridgeClient)
        client.pose_sink = None

        def send_command(index=index, **kwargs):
            commands[index] = sim.parse_command(_command_payload(**kwargs))
            return 1

        client.transport = SimpleNamespace(send_command=send_command)

        def read_next(timeout, binding=binding):
            q, dq, quat, gyro, _ = sim._robot_state(binding)
            return replace(
                state(),
                q_lab=q,
                dq_lab=dq,
                quat_wxyz=quat,
                gyro=gyro,
                buttons=sim._button_snapshot.copy(),
                state_receive_time_ns=sim._snapshot_id + 1,
            )

        client.read_next = read_next
        clients.append(client)
    sessions = [
        RobotSession(
            RobotSessionConfig(
                robot_id=name,
                udp={},
                lower=np.full(29, -3),
                upper=np.full(29, 3),
                kp=np.ones(29),
                kd=np.ones(29),
                max_target_delta=1.0,
                torque_limit=sim.torque_limits,
            ),
            client=client,
        )
        for name, client in zip(("a", "b"), clients)
    ]
    coordinator = InteractiveDualCoordinator(
        *sessions,
        provider,
        Policy(),
        enable_a=True,
        enable_b=True,
        loco_modes=tuple(
            LocoModePolicy(assets, list(POLICY_JOINT_NAMES)) for _ in range(2)
        ),
        default_command=load_default_command(assets),
    )
    try:
        for tick_index in range(130):
            if tick_index == 2:
                sim.key_callback(ord("s"))
            if tick_index == 25:
                sim.key_callback(ord("b"))
            sim.publish_state_pair(sim._snapshot_id + 1)
            provider.ingest_bridge_state({"dual_sim_pose": sim.pose_payload()})
            result = coordinator.step()
            assert result.ok
            sim.step_policy_interval(tuple(commands))
            sim._snapshot_id += 1
            assert np.all(np.isfinite(sim.data.qpos))
        assert coordinator.state == DeploymentState.LOCO_STANDING
        assert sim._roots_released and not sim._task_started
        assert sim.task_termination_reason() is None
        for binding in sim.bindings:
            assert sim.data.qpos[binding.root_qpos + 2] > 0.6
            # Both roots remain upright through >2 s of unsupported standing.
            quaternion = sim.data.qpos[binding.root_qpos + 3 : binding.root_qpos + 7]
            assert 1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2) > 0.9
    finally:
        provider.stop()
        sim.close()


def test_retransmitted_snapshot_does_not_advance_policy_or_limiter(runtime):
    coordinator, clients, policy, locos = runtime
    for stamp, key in enumerate(("start", "B", "A"), start=1):
        for client in clients:
            client.stamp = stamp
        tick(runtime, key)
    assert policy.calls == 1
    originals = [client.sent[-1] for client in clients]
    assert tick(runtime, "A").reason == "simulation_snapshot_retry"
    assert policy.calls == 1
    for client, (original_command, original_kwargs) in zip(clients, originals):
        assert client.sent[-1][0] is original_command
        assert client.sent[-1][1]["sim_control"] == original_kwargs["sim_control"]


def test_mixed_snapshot_pair_fails_closed(runtime):
    runtime[1][0].stamp = 10
    runtime[1][1].stamp = 9
    result = tick(runtime, "start")
    assert not result.ok and result.reason == "mismatched_simulation_snapshots"
    assert all(client.sent[-1][1]["enable"] == 0 for client in runtime[1])
