from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest

from dual_runtime.constants import KEY_BODY_NAMES
from dual_runtime.observation import build_pelvis_residual_observation
from dual_runtime.onboard_policy import OnboardPolicy
from dual_runtime.dual_pose_provider import DualPoseSnapshot
from dual_runtime.reference import ReferenceFrame
from omnicontact.contracts import RobotPose, ObjectPose
from omnicontact.runtime import BridgeState
from omnicontact.reference.math_wxyz import quat_apply_batch, quat_mul_left_batch


def state(q=None):
    return BridgeState(np.zeros(29) if q is None else q, np.zeros(29),
                       np.array([1., 0, 0, 0]), np.array([.1, .2, .3]), {},
                       None, 1, time.monotonic_ns())


def test_pelvis_interactions_and_reference_torso_local_angular_velocity():
    quats = np.tile([1., 0, 0, 0], (14, 1))
    quats[KEY_BODY_NAMES.index("torso_link")] = [2**-.5, 2**-.5, 0, 0]
    ref = ReferenceFrame(np.zeros(29), np.zeros(29), np.zeros((14, 3)), quats,
                         np.array([0., 1, 0]), np.zeros(3), np.array([1., 0, 0, 0]))
    own = RobotPose([1, 2, 3], [0, 0, 2**-.5, 2**-.5], 1.)
    partner = RobotPose([3, 3, 3], [0, 0, 0, 1], 1.)
    box = ObjectPose([2, 4, 3], [0, 0, 0, 1], [.5, .15, .15], 1.)
    args = dict(state=state(), reference=ref, own_pelvis=own, partner_pelvis=partner,
                object_pose=box, scalebfm_target=np.zeros(29),
                previous_residual=np.zeros(29), default_q=np.zeros(29))
    obs = build_pelvis_residual_observation(**args)
    assert obs.shape == (201,)
    np.testing.assert_allclose(obs[61:64], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[154:157], [1, -2, 0], atol=1e-6)
    np.testing.assert_allclose(obs[192:195], [2, -1, 0], atol=1e-6)
    np.testing.assert_allclose(obs[157:163], [0, 1, -1, 0, 0, 0], atol=1e-6)
    world = build_pelvis_residual_observation(**args, anchor_angular_velocity_frame="world")
    np.testing.assert_allclose(world[61:64], [0, 1, 0])
    # Common world yaw rotates the reference torso AND world angular velocity.
    rotation = np.array([2**-.5, 0, 0, 2**-.5])
    ref2 = replace(ref, body_quat_wxyz=quat_mul_left_batch(rotation, ref.body_quat_wxyz),
                   anchor_ang_vel_w=quat_apply_batch(rotation, ref.anchor_ang_vel_w[None])[0])
    local2 = build_pelvis_residual_observation(**{**args, "reference": ref2})
    np.testing.assert_allclose(local2[61:64], obs[61:64], atol=1e-6)


@pytest.fixture(scope="module")
def policies():
    root = Path(__file__).resolve().parents[1] / "config/g1/dual_policy_artifacts_contact_v2_8192"
    kwargs = dict(scalebfm_checkpoint=root/"scalebfm_model.pt",
                  scalebfm_metadata=root/"scalebfm_metadata.json",
                  scalebfm_mode_table=root/"scalebfm_mode_table.pt",
                  residual_checkpoint=root/"residual_actor.pt",
                  reference_bundle=root/"reference_bundle.npz",
                  kinematics_xml=root/"g1_29dof_scalebfm.xml",
                  reference_alignment="motion_world", control_mode=2, torch_num_threads=2)
    return [OnboardPolicy(robot_id=side, **kwargs) for side in ("a", "b")]


def test_single_policy_matches_batched_networks_and_never_reads_partner_fk(policies):
    from copy import copy
    from dual_runtime.scalebfm_policy import ScaleBFMHistory
    refs = policies[0].reference.frame(1)
    poses = [RobotPose(r.body_pos_w[0], r.body_quat_wxyz[0][[1, 2, 3, 0]], time.monotonic()) for r in refs]
    snapshot = DualPoseSnapshot(*poses, ObjectPose(refs[0].object_pos_w,
        refs[0].object_quat_wxyz[[1, 2, 3, 0]], policies[0].reference.box_half_extents, time.monotonic()))
    states = [state(r.joint_pos) for r in refs]
    histories = (ScaleBFMHistory(), ScaleBFMHistory())
    previous = np.zeros((2, 29), dtype=np.float32)
    previous_residual = np.zeros_like(previous)
    for p in policies:
        p.initialized = True
        p.reset_rollout()
    base = policies[0]
    central = copy(base)
    central.histories = (ScaleBFMHistory(), ScaleBFMHistory())
    central.previous_residual = np.zeros((2, 29), dtype=np.float32)
    central.previous_executed_action = np.zeros((2, 29), dtype=np.float32)
    central.reset_rollout()
    for frame in range(1, 5):
        live = [p.kinematics[p.index].forward(s.q_lab, pose) for p, s, pose in zip(policies, states, poses)]
        for i, s in enumerate(states):
            histories[i].update(s.quat_wxyz, s.gyro, s.q_lab, s.dq_lab, previous[i])
        pos, quat = base.reference.future_key_bodies(frame, base.time_offsets)
        target, action = base.scalebfm.infer_batch(histories, pos, quat,
            np.stack([v.body_pos_w for v in live]), np.stack([v.body_quat_wxyz for v in live]),
            control_mode=base.control_mode, time_offsets=base.time_offsets)
        observations = np.stack([build_pelvis_residual_observation(
            state=states[i], reference=base.reference.frame(frame)[i], own_pelvis=poses[i],
            partner_pelvis=poses[1-i], object_pose=snapshot.object, scalebfm_target=target[i],
            previous_residual=previous_residual[i], default_q=base.default_q) for i in range(2)])
        residual = base.residual.infer(observations)
        expected = target + base.residual_scale*residual
        central_step = central.compute(tuple(states), snapshot)
        np.testing.assert_allclose(central_step.observations, observations, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(central_step.targets, expected, atol=2e-5, rtol=1e-5)
        for p, s in zip(policies, states):
            original = p.kinematics
            class ForbiddenFK:
                def forward(self, *args):
                    raise AssertionError("partner FK must not run onboard")
            replacement = list(original)
            replacement[1-p.index] = ForbiddenFK()
            p.kinematics = replacement
            try:
                step = p.compute_single(s, snapshot)
            finally:
                p.kinematics = original
            np.testing.assert_allclose(step.target, expected[p.index], atol=2e-4, rtol=1e-4)
            np.testing.assert_allclose(step.observation, observations[p.index], atol=2e-4, rtol=1e-4)
            assert step.frame == frame
        previous = action + base.residual_scale*residual/base.scalebfm.action_scale
        previous_residual = residual


def test_legacy_manifest_rejected_for_onboard():
    from dual_runtime.onboard_config import load_artifacts
    root = Path(__file__).resolve().parents[1] / "config/g1/dual_policy_artifacts_contact_v2_8192"
    with pytest.raises(ValueError, match="pelvis/reference-anchor"):
        load_artifacts(root)
