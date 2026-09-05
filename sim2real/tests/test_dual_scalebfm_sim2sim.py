import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from dataclasses import replace
from dual_runtime.constants import POLICY_JOINT_NAMES
from dual_runtime.sim_control import load_default_command

from dual_runtime.sim_pose_provider import DualSimulationPoseProvider
from dual_scalebfm_sim2sim import DualScaleBFMSim2Sim, SimCommand


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/g1/dual_scalebfm_residual.yaml"


def configure_test_scene(tmp_path, monkeypatch):
    """Exercise real MuJoCo without depending on untracked policy checkpoints."""
    raw = yaml.safe_load(CONFIG.read_text())
    sim = raw["simulation"]
    sim["xml_path"] = str((CONFIG.parent / sim["xml_path"]).resolve())
    raw.setdefault("control", {})["standing_asset_dir"] = str(CONFIG.parent / "omnicontact")
    sim["joint_dynamics_xml"] = str(
        (CONFIG.parent / sim["joint_dynamics_xml"]).resolve()
    )
    sim["torque_limits"] = yaml.safe_load(
        (CONFIG.parent / "bridge_omnicontact.yaml").read_text()
    )["torque_limits"]
    reference = tmp_path / "reference.npz"
    arrays = {
        "training_joint_order": np.asarray(POLICY_JOINT_NAMES),
        "training_box_half_extents": np.asarray(sim["box_half_extents"]),
        "training_object_body_pos_w": np.tile([0.0, 0.5, 0.15], (4, 1)),
        "training_object_body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
    }
    for index in range(2):
        arrays[f"training_robot_{index}_body_pos_w"] = np.tile(
            [0.0, index * 1.2, 0.8], (4, 1, 1)
        )
        arrays[f"training_robot_{index}_body_quat_w"] = np.tile(
            [1.0, 0.0, 0.0, 0.0], (4, 1, 1)
        )
        arrays[f"training_robot_{index}_joint_pos"] = np.zeros((4, 29))
    np.savez(reference, **arrays)
    raw["artifacts"]["directory"] = str(tmp_path)
    raw["artifacts"]["reference_bundle"] = reference.name
    config = tmp_path / "sim.yaml"
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(__import__(__name__), "CONFIG", config)


@pytest.fixture(autouse=True)
def isolated_reference(tmp_path, monkeypatch):
    configure_test_scene(tmp_path, monkeypatch)


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


def command(
    sim,
    robot_index,
    *,
    enable,
    state_time_ns=123,
    offset=0.0,
    phase="default_pose",
    frame=-1,
):
    binding = sim.bindings[robot_index]
    q = sim.data.qpos[binding.joint_qpos].copy()
    return SimCommand(
        q_des=q + offset,
        qd_des=np.zeros(29),
        kp=np.full(29, 100.0),
        kd=np.full(29, 2.0),
        enable=enable,
        state_time_ns=state_time_ns,
        phase=phase,
        frame=frame,
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
            sim.data.ctrl[b.actuators], -command(sim, 1, enable=0).kd * 0.5, atol=1e-6
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
        sim._snapshot_id = 10000
        commands = tuple(replace(c, phase="loco_standing") for c in commands)
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
        sim._phase = "executing"
        sim._policy_frame = sim.start_frame
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
        np.testing.assert_allclose(snapshot.object.half_extents, sim.box_half_extents)
        assert snapshot.robot_a.position_w[1] < snapshot.robot_b.position_w[1]
    finally:
        provider.stop()
        sim.close()


def test_startup_default_pose_and_zero_torque_do_not_drift():
    sim = make_sim()
    try:
        defaults = load_default_command(ROOT / "config/g1/omnicontact")
        for binding in sim.bindings:
            np.testing.assert_allclose(
                sim.data.qpos[binding.joint_qpos], defaults.target_pos
            )
        initial = sim.data.qpos.copy()
        sim.data.qvel[:] = 0.2
        for _ in range(10):
            sim.step_policy_interval(
                tuple(command(sim, i, enable=1, phase="zero_torque") for i in range(2))
            )
        np.testing.assert_array_equal(sim.data.qpos, initial)
        assert sim.data.time == 0
        assert sim.task_termination_reason() is None
    finally:
        sim.close()


def test_asymmetric_release_is_rejected_before_physics():
    sim = make_sim()
    try:
        with pytest.raises(RuntimeError, match="disagree"):
            sim.step_policy_interval(
                (
                    command(sim, 0, enable=1, phase="loco_standing"),
                    command(sim, 1, enable=1),
                )
            )
        assert not sim._roots_released
        assert sim.data.time == 0
    finally:
        sim.close()


def test_waiting_ticks_do_not_advance_task_frame_or_termination():
    sim = make_sim()
    try:
        sim._snapshot_id = 100000
        sim.data.qpos[sim.box_qpos + 2] += 0.5
        mujoco.mj_forward(sim.model, sim.data)
        for phase in ("zero_torque", "default_pose", "loco_standing", "stopped"):
            sim._phase = phase
            assert sim.task_termination_reason() is None
        sim._phase = "executing"
        sim._policy_frame = 2
        assert sim._reference_frame() == 2
        assert sim.task_termination_reason() is not None
        # The controller's aligned reference is authoritative.
        sim._command_reference_position = tuple(sim.data.xpos[sim.box_body])
        assert sim.task_termination_reason() is None
    finally:
        sim.close()


def test_keyboard_snapshot_is_atomic_across_robots_and_retries():
    sim = make_sim()
    try:
        sim.key_callback(ord("s"))
        sim.publish_state_pair(10)
        sim.key_callback(ord("b"))
        sim.publish_state_pair(10)
        for transport in sim.transports:
            assert transport.states[-1]["buttons"] == transport.states[-2]["buttons"]
            assert transport.states[-1]["buttons"]["start"]
            assert not transport.states[-1]["buttons"]["B"]
        sim._snapshot_id += 1
        sim.publish_state_pair(20)
        assert not any(sim.transports[0].states[-1]["buttons"].values())
        sim._snapshot_id += 1
        sim.publish_state_pair(30)
        assert sim.transports[0].states[-1]["buttons"]["B"]
        sim.key_callback(ord("a"))
        sim.key_callback(ord("x"))
        sim._snapshot_id += 1
        sim.publish_state_pair(40)
        assert sim.transports[0].states[-1]["buttons"]["stop"]
        assert not sim.transports[0].states[-1]["buttons"]["A"]
    finally:
        sim.close()


def test_joint_passive_dynamics_match_single_g1_for_both_robots():
    sim = make_sim()
    try:
        source = mujoco.MjModel.from_xml_path(
            str(ROOT / "config/g1/assets/g1_29dof.xml")
        )
        dofs = [
            source.jnt_dofadr[
                mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in POLICY_JOINT_NAMES
        ]
        for binding in sim.bindings:
            for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
                np.testing.assert_array_equal(
                    getattr(sim.model, field)[binding.joint_dof],
                    getattr(source, field)[dofs],
                )
    finally:
        sim.close()


def test_reference_box_dimensions_update_collision_and_inertia():
    raw = yaml.safe_load(CONFIG.read_text())
    half = np.array([.48, .15, .15])
    raw['simulation']['box_half_extents'] = half.tolist()
    reference = Path(raw['artifacts']['directory']) / raw['artifacts']['reference_bundle']
    with np.load(reference) as data:
        arrays = {key:data[key] for key in data.files}
    arrays['training_box_half_extents'] = half
    np.savez(reference, **arrays)
    CONFIG.write_text(yaml.safe_dump(raw))
    sim = make_sim()
    try:
        geom = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, 'box_collision')
        np.testing.assert_allclose(sim.model.geom_size[geom], half)
        mass = sim.model.body_mass[sim.box_body]
        np.testing.assert_allclose(sim.model.body_inertia[sim.box_body], mass / 3 * np.array([
            half[1]**2 + half[2]**2,half[0]**2 + half[2]**2,half[0]**2 + half[1]**2]))
    finally:
        sim.close()
