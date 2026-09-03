import time
from pathlib import Path

import mujoco
import numpy as np

from dual_runtime.sim_pose_provider import DualSimulationPoseProvider
from dual_scalebfm_sim2sim import DualScaleBFMSim2Sim, SimCommand


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/g1/dual_scalebfm_residual.yaml"


class FakeLowTransport:
    def __init__(self):
        self.states = []
        self.closed = False

    def send_state(self, **values):
        self.states.append(values)
        return len(self.states) - 1

    def close(self):
        self.closed = True


def make_sim():
    transports = (FakeLowTransport(), FakeLowTransport())
    return DualScaleBFMSim2Sim(CONFIG, headless=True, transports=transports)


def command(sim, robot_index, *, enable, state_time_ns=123, offset=0.0):
    binding = sim.bindings[robot_index]
    q = sim.data.qpos[binding.joint_qpos].copy()
    return SimCommand(
        q_des=q + offset,
        qd_des=np.zeros(29),
        kp=np.full(29, 100.0),
        kd=np.full(29, 2.0),
        enable=enable,
        state_time_ns=state_time_ns,
    )


def test_dual_physics_model_has_independent_robot_addresses():
    sim = make_sim()
    try:
        assert sim.model.nu == 58
        assert not np.intersect1d(
            sim.bindings[0].joint_qpos, sim.bindings[1].joint_qpos
        ).size
        assert not np.intersect1d(
            sim.bindings[0].actuators, sim.bindings[1].actuators
        ).size
        for name in ("a_floating_base_joint", "b_floating_base_joint", "box_joint"):
            assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        for prefix in ("a_", "b_"):
            for side in ("left", "right"):
                assert (
                    mujoco.mj_name2id(
                        sim.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"{prefix}{side}_hand_collision",
                    )
                    >= 0
                )
    finally:
        sim.close()


def test_state_pair_contains_one_atomic_dual_pose_snapshot():
    sim = make_sim()
    try:
        sim.publish_state_pair(987654321)
        a, b = (transport.states[-1] for transport in sim.transports)
        assert a["state_receive_time_ns"] == b["state_receive_time_ns"] == 987654321
        assert a["extra_state"]["dual_sim_pose"]["snapshot_id"] == 0
        assert b["extra_state"]["dual_sim_pose"]["snapshot_id"] == 0
        np.testing.assert_allclose(
            a["extra_state"]["dual_sim_pose"]["robots"][1]["position_w"],
            b["extra_state"]["dual_sim_pose"]["robots"][1]["position_w"],
        )
    finally:
        sim.close()


def test_bridge_gyro_uses_body_frame_imu_sensor():
    sim = make_sim()
    try:
        binding = sim.bindings[0]
        assert binding.gyro_sensor is not None
        sim.data.qvel[binding.root_dof + 3 : binding.root_dof + 6] = [1.0, 0.0, 0.0]
        mujoco.mj_forward(sim.model, sim.data)
        gyro = sim._robot_state(binding)[3]
        address = binding.gyro_sensor[0]
        np.testing.assert_allclose(gyro, sim.data.sensordata[address : address + 3])
    finally:
        sim.close()


def test_lockstep_requires_both_commands_for_the_exact_state():
    sim = make_sim()
    try:
        sim._commands[0] = command(sim, 0, enable=1, state_time_ns=100)
        sim._commands[1] = command(sim, 1, enable=1, state_time_ns=99)
        assert not sim.command_pair_ready(100)
        sim._commands[1] = command(sim, 1, enable=1, state_time_ns=100)
        assert sim.command_pair_ready(100)
    finally:
        sim.close()


def test_enable_zero_is_damping_and_enable_one_is_pd():
    sim = make_sim()
    try:
        a, b = sim.bindings
        sim.data.qvel[a.joint_dof] = 0.0
        sim.data.qvel[b.joint_dof] = 0.5
        sim.apply_commands(
            (
                command(sim, 0, enable=1, offset=0.1),
                command(sim, 1, enable=0, offset=1.0),
            )
        )
        np.testing.assert_allclose(
            sim.data.ctrl[a.actuators], np.minimum(10.0, sim.torque_limits), atol=1e-6
        )
        np.testing.assert_allclose(
            sim.data.ctrl[b.actuators], -sim.disabled_damping_kd * 0.5, atol=1e-6
        )
    finally:
        sim.close()


def test_pd_is_recomputed_from_latest_state_on_every_physics_substep():
    sim = make_sim()
    try:
        commands = (
            command(sim, 0, enable=1, offset=0.002),
            command(sim, 1, enable=1, offset=-0.002),
        )
        applied = []
        original_apply_commands = sim.apply_commands

        def record_apply(current_commands):
            original_apply_commands(current_commands)
            applied.append(sim.data.ctrl.copy())

        sim.apply_commands = record_apply
        sim.step_policy_interval(commands)

        assert len(applied) == sim.decimation
        assert not np.allclose(applied[0], applied[-1], rtol=0.0, atol=1e-9)
    finally:
        sim.close()


def test_roots_are_locked_only_during_default_pose_phase():
    sim = make_sim()
    try:
        initial = [
            sim.data.qpos[binding.root_qpos : binding.root_qpos + 7].copy()
            for binding in sim.bindings
        ]
        initial_box = sim.data.qpos[sim.box_qpos : sim.box_qpos + 7].copy()
        sim.data.qvel[sim.box_dof + 2] = 1.0
        commands = (
            command(sim, 0, enable=1),
            command(sim, 1, enable=1),
        )
        sim.step_policy_interval(commands)
        for binding, expected in zip(sim.bindings, initial):
            np.testing.assert_allclose(
                sim.data.qpos[binding.root_qpos : binding.root_qpos + 7], expected
            )
        np.testing.assert_allclose(
            sim.data.qpos[sim.box_qpos : sim.box_qpos + 7], initial_box
        )
        sim._snapshot_id = sim.default_pose_steps
        sim.data.qvel[sim.bindings[0].root_dof + 2] = 1.0
        before = sim.data.qpos[sim.bindings[0].root_qpos + 2]
        sim.step_policy_interval(commands)
        after = sim.data.qpos[sim.bindings[0].root_qpos + 2]
        assert not np.isclose(after, before)
    finally:
        sim.close()


def test_training_object_height_termination_stops_failed_rollout():
    sim = make_sim()
    try:
        sim._snapshot_id = sim.default_pose_steps
        sim.data.qpos[sim.box_qpos + 2] += sim.object_position_z_error_m + 0.01
        mujoco.mj_forward(sim.model, sim.data)
        reason = sim.task_termination_reason()
        assert reason is not None
        assert reason.startswith("object.position_z")
    finally:
        sim.close()


def test_sim_pose_provider_converts_a_single_atomic_snapshot():
    sim = make_sim()
    provider = DualSimulationPoseProvider()
    try:
        provider.start()
        provider.ingest_bridge_state({"dual_sim_pose": sim.pose_payload()})
        assert provider.wait_until_ready(0.01)
        snapshot = provider.get_snapshot()
        assert snapshot is not None
        assert 0.0 <= time.monotonic() - snapshot.robot_a.stamp_s < 0.1
        np.testing.assert_allclose(
            snapshot.object.half_extents, [0.30, 0.15, 0.15]
        )
        assert snapshot.robot_a.position_w[1] < snapshot.robot_b.position_w[1]
    finally:
        provider.stop()
        sim.close()
