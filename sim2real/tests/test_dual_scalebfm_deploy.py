import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from dual_runtime.constants import KEY_BODY_NAMES, POLICY_JOINT_NAMES
from dual_runtime.dual_pose_provider import DualPoseSnapshot
from dual_runtime.kinematics import LiveKinematics
from dual_runtime.observation import build_residual_observation
from dual_runtime.policy_coordinator import DeploymentState, DualPolicyCoordinator
from dual_runtime.reference import DualReferenceBundle, ReferenceFrame
from dual_runtime.robot_session import RobotSession, RobotSessionConfig
from dual_runtime.vive_dual_pose import (
    DualViveDeploymentConfig,
    DualVivePoseProvider,
)
from omnicontact.contracts import ObjectPose, PDCommand, RobotPose
from omnicontact.perception.openvr_tracker import ViveSample
from omnicontact.perception.vive_pose import RigidTransform
from omnicontact.runtime import BridgeState


def state(q=None, dq=None):
    return BridgeState(
        q_lab=np.zeros(29, dtype=np.float32)
        if q is None
        else np.asarray(q, np.float32),
        dq_lab=np.zeros(29, dtype=np.float32)
        if dq is None
        else np.asarray(dq, np.float32),
        quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        buttons={},
        state_receive_time_ns=None,
        packet_seq=1,
        packet_arrival_ns=time.monotonic_ns(),
    )


def write_reference(path: Path):
    frames = 8
    bodies = len(KEY_BODY_NAMES)
    positions = np.zeros((frames, bodies, 3), dtype=np.float32)
    quaternions = np.zeros((frames, bodies, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    positions[:, :, 2] = 0.8
    positions_b = positions.copy()
    positions_b[:, :, 1] = 1.0
    zeros_joint = np.zeros((frames, 29), dtype=np.float32)
    zeros_body = np.zeros((frames, bodies, 3), dtype=np.float32)
    np.savez(
        path,
        fps=np.float32(50),
        training_robot_count=np.int32(2),
        training_joint_order=np.asarray(POLICY_JOINT_NAMES),
        training_body_order=np.asarray(KEY_BODY_NAMES),
        training_robot_0_joint_pos=zeros_joint,
        training_robot_0_joint_vel=zeros_joint,
        training_robot_0_body_pos_w=positions,
        training_robot_0_body_quat_w=quaternions,
        training_robot_0_body_ang_vel_w=zeros_body,
        training_robot_1_joint_pos=zeros_joint,
        training_robot_1_joint_vel=zeros_joint,
        training_robot_1_body_pos_w=positions_b,
        training_robot_1_body_quat_w=quaternions,
        training_robot_1_body_ang_vel_w=zeros_body,
        training_object_body_pos_w=np.tile([0.0, 0.5, 0.5], (frames, 1)),
        training_object_body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        training_box_half_extents=np.asarray([0.3, 0.15, 0.15], np.float32),
    )


def test_reference_alignment_moves_both_robots_and_object(tmp_path):
    path = tmp_path / "reference.npz"
    write_reference(path)
    bundle = DualReferenceBundle(path)
    bundle.align_to_robot_a(RobotPose([2, 3, 0.8], [0, 0, 0, 1], 1.0))
    robot_a, robot_b = bundle.frame(1)
    np.testing.assert_allclose(robot_a.body_pos_w[0], [2, 3, 0.8], atol=1e-6)
    np.testing.assert_allclose(robot_b.body_pos_w[0], [2, 4, 0.8], atol=1e-6)
    np.testing.assert_allclose(robot_a.object_pos_w, [2, 3.5, 0.5], atol=1e-6)


def test_observation_matches_201_dimension_contract():
    body_pos = np.zeros((14, 3), dtype=np.float32)
    body_quat = np.zeros((14, 4), dtype=np.float32)
    body_quat[:, 0] = 1.0
    live_a = LiveKinematics(body_pos, body_quat, np.zeros(3), body_quat[0])
    live_b = LiveKinematics(body_pos, body_quat, np.array([1, 0, 0]), body_quat[0])
    reference = ReferenceFrame(
        np.zeros(29),
        np.zeros(29),
        body_pos,
        body_quat,
        np.zeros(3),
        np.zeros(3),
        np.array([1, 0, 0, 0]),
    )
    object_pose = ObjectPose([0.5, 0, 0.3], [0, 0, 0, 1], [0.3, 0.15, 0.15], 1.0)
    observation = build_residual_observation(
        state=state(),
        reference=reference,
        live=live_a,
        partner_live=live_b,
        object_pose=object_pose,
        scalebfm_target=np.zeros(29),
        previous_residual=np.zeros(29),
        default_q=np.zeros(29),
    )
    assert observation.shape == (201,)
    np.testing.assert_allclose(observation[58:61], [0, 0, -1], atol=1e-6)


class FakeClient:
    def __init__(self):
        self.sent = []

    def send(self, command, *, enable, state):
        self.sent.append((command, enable, state))

    def close(self):
        pass


def test_robot_session_limits_estimated_pd_torque():
    client = FakeClient()
    session = RobotSession(
        RobotSessionConfig(
            robot_id="a",
            udp={},
            lower=np.full(29, -3.0),
            upper=np.full(29, 3.0),
            kp=np.full(29, 100.0),
            kd=np.zeros(29),
            torque_limit=np.full(29, 10.0),
            max_target_delta=1.0,
        ),
        client=client,
    )
    command = session.send_target(np.ones(29), state(), enable=1)
    np.testing.assert_allclose(command.target_pos, 0.1, atol=1e-6)
    np.testing.assert_allclose(
        100.0 * (command.target_pos - np.zeros(29)), 10.0, atol=1e-5
    )


class FakeTrackerReader:
    def __init__(self):
        self.serial_to_index = {"A": 1, "B": 2, "BOX": 3}

    def start(self):
        return self.serial_to_index

    def stop(self):
        pass

    def refresh_devices(self):
        return self.serial_to_index

    def read_all(self, serials):
        return {serial: sample(index) for index, serial in enumerate(serials)}


def sample(x):
    return ViveSample(1, 2, float(x), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_three_tracker_provider_publishes_one_atomic_snapshot():
    identity = RigidTransform([0, 0, 0], [0, 0, 0, 1])
    config = DualViveDeploymentConfig(
        "A",
        "B",
        "BOX",
        identity,
        identity,
        identity,
        identity,
        [0.3, 0.15, 0.15],
        True,
    )
    provider = DualVivePoseProvider(config, reader=FakeTrackerReader())
    assert provider.update_once()
    snapshot = provider.get_snapshot()
    assert isinstance(snapshot, DualPoseSnapshot)
    np.testing.assert_allclose(snapshot.robot_a.position_w, [0, 0, 1])
    np.testing.assert_allclose(snapshot.robot_b.position_w, [1, 0, 1])
    np.testing.assert_allclose(snapshot.object.position_w, [2, 0, 1])


def test_residual_actor_uses_frozen_scaler_and_deterministic_mean(tmp_path):
    torch = pytest.importorskip("torch")
    from dual_runtime.residual_policy import ResidualPolicy

    modules = {}
    for agent in ("robot_0", "robot_1"):
        policy = {"log_std_parameter": torch.zeros(29)}
        widths = (201, 512, 256, 128, 29)
        for index, (input_width, output_width) in enumerate(pairwise(widths)):
            policy[f"net_container.{2 * index}.weight"] = torch.zeros(
                output_width, input_width
            )
            policy[f"net_container.{2 * index}.bias"] = torch.zeros(output_width)
        policy["net_container.6.bias"].fill_(0.25)
        modules[agent] = {
            "policy": policy,
            "state_preprocessor": {
                "running_mean": torch.zeros(201, dtype=torch.float64),
                "running_variance": torch.ones(201, dtype=torch.float64),
                "current_count": torch.tensor(1.0, dtype=torch.float64),
            },
        }
    checkpoint = tmp_path / "actor.pt"
    torch.save(modules, checkpoint)
    runtime = ResidualPolicy(checkpoint)
    np.testing.assert_allclose(runtime.infer(np.ones((2, 201))), 0.25)


class FakeSession:
    def __init__(self):
        self.last_state = state()
        self.sent = []

    def read(self, timeout_s):
        self.last_state = state()
        return self.last_state

    def send_target(self, target, current, *, enable):
        self.sent.append(("target", np.asarray(target).copy(), enable))
        return PDCommand(target)

    def send_hold(self, current, *, enable=0):
        self.sent.append(("hold", current.q_lab.copy(), enable))

    def close(self):
        pass


class FakePoseProvider:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def get_snapshot(self):
        return self.snapshot


class FakeCoupledPolicy:
    def __init__(self):
        self.default_q = np.full(29, 0.1, dtype=np.float32)
        self.initialized = False

    def initialize(self, snapshot, **limits):
        self.initialized = True
        return {"partner_position_error_m": 0.0}

    def reset_rollout(self):
        pass

    def compute(self, states, snapshot):
        from types import SimpleNamespace

        return SimpleNamespace(
            targets=np.stack((np.full(29, 0.2), np.full(29, -0.2))),
            scalebfm_targets=np.zeros((2, 29)),
            residuals=np.zeros((2, 29)),
            observations=np.zeros((2, 201)),
            frame=1,
            complete=False,
            inference_time_s=0.001,
        )


def fresh_snapshot():
    stamp = time.monotonic()
    return DualPoseSnapshot(
        RobotPose([0, 0, 0.8], [0, 0, 0, 1], stamp),
        RobotPose([0, 1, 0.8], [0, 0, 0, 1], stamp),
        ObjectPose([0, 0.5, 0.3], [0, 0, 0, 1], [0.3, 0.15, 0.15], stamp),
    )


def test_production_coordinator_routes_default_then_coupled_targets():
    robot_a, robot_b = FakeSession(), FakeSession()
    coordinator = DualPolicyCoordinator(
        robot_a,
        robot_b,
        FakePoseProvider(fresh_snapshot()),
        FakeCoupledPolicy(),
        enable_a=True,
        enable_b=False,
        default_pose_ticks=1,
    )
    first = coordinator.step()
    assert first.ok and first.state == DeploymentState.EXECUTING
    np.testing.assert_allclose(robot_a.sent[-1][1], 0.1)
    second = coordinator.step()
    assert second.ok and second.reason == "policy_target"
    np.testing.assert_allclose(robot_a.sent[-1][1], 0.2)
    np.testing.assert_allclose(robot_b.sent[-1][1], -0.2)
    assert robot_a.sent[-1][2] == 1
    assert robot_b.sent[-1][2] == 0
