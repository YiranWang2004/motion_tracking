"""Diagnostic stages survive recording without changing sent commands."""
from types import SimpleNamespace

import numpy as np
import pytest

from dual_runtime.deployment_recorder import DualDeploymentRecorder
from dual_runtime.policy_coordinator import CoordinatorResult, DeploymentState
from dual_runtime.robot_session import RobotSession, RobotSessionConfig
from omnicontact.contracts import PDCommand
from test_dual_scalebfm_deploy import fresh_snapshot, state
from test_dual_sim_control import Client


def session(name="a"):
    return RobotSession(RobotSessionConfig(
        robot_id=name, udp={}, lower=np.full(29, -1), upper=np.ones(29),
        kp=np.full(29, 100), kd=np.ones(29), torque_limit=np.full(29, 10),
        max_target_delta=.2,
    ), Client())


def test_target_stages_and_positive_negative_torque_clipping():
    robot = session()
    desired = np.zeros(29, dtype=np.float32)
    desired[:3] = [2, -2, .05]
    command = robot.send_target(desired, state(), enable=1)
    info = robot.last_command_diagnostics
    np.testing.assert_allclose(info["command_requested_target"][:3], [2, -2, .05])
    np.testing.assert_allclose(info["command_pre_limit_target"][:3], [2, -2, .05])
    np.testing.assert_allclose(info["command_post_target_limit"][:3], [.2, -.2, .05])
    np.testing.assert_allclose(info["estimated_torque_pre_limit"][:3], [20, -20, 5])
    np.testing.assert_allclose(info["estimated_torque_clipped"][:3], [10, -10, 5])
    np.testing.assert_array_equal(info["torque_limit_triggered"][:3], [True, True, False])
    np.testing.assert_allclose(command.target_pos[:3], [.1, -.1, .05])
    assert robot.client.sent[-1][0] is command
    desired.fill(0)
    assert info["command_requested_target"][0] == 2


@pytest.mark.parametrize("enable", [0, 1])
def test_disabled_and_bypassed_torque_limit(enable):
    robot = session()
    robot.send_pd(PDCommand(np.ones(29), np.full(29, 100), np.ones(29)),
                  state(), enable=enable, limit_estimated_torque=False)
    info = robot.last_command_diagnostics
    assert not info["torque_limit_enabled"]
    assert not info["torque_limit_triggered"].any()
    assert np.isnan(info["estimated_torque_pre_limit"]).all()
    np.testing.assert_array_equal(info["command_requested_target"], 1)
    np.testing.assert_array_equal(info["command_pre_limit_target"], enable)


def test_standing_record_round_trip_and_fault_clears_old_diagnostics(tmp_path):
    robots = [session("a"), session("b")]
    for index, robot in enumerate(robots):
        robot.last_state = state()
        robot.send_target(np.full(29, .5 if index == 0 else -.5),
                          robot.last_state, enable=1)
    coordinator = SimpleNamespace(robots=robots,
        pose_provider=SimpleNamespace(get_snapshot=fresh_snapshot))
    recorder = DualDeploymentRecorder(tmp_path, {})
    recorder.record(coordinator, CoordinatorResult(True, DeploymentState.SCALEBFM_STANDING,
                                                  "scalebfm_default_pose_standing"))
    for robot in robots:
        robot.send_hold(robot.last_state)
        assert not robot.last_command_diagnostics
    recorder.record(coordinator, CoordinatorResult(False, DeploymentState.FAULT, "pose_loss"))
    with np.load(recorder.save(), allow_pickle=False) as log:
        assert log["scalebfm_target"].shape == (2, 2, 29)
        np.testing.assert_allclose(log["scalebfm_target"][0, :, 0], [.5, -.5])
        np.testing.assert_allclose(log["command_post_target_limit"][0, :, 0], [.2, -.2])
        np.testing.assert_allclose(log["command_target"][0, :, 0], [.1, -.1])
        assert log["torque_limit_triggered"][0].all()
        assert log["command_diagnostics_valid"][0].all()
        assert not log["command_diagnostics_valid"][1].any()
        assert not log["torque_limit_triggered"][1].any()
        assert np.isnan(log["scalebfm_target"][1]).all()
        assert np.isnan(log["command_pre_limit_target"][1]).all()
