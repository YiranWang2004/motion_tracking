"""Base-policy tests require neither a residual checkpoint nor a robot connection."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_dual_scalebfm_deploy import write_reference, state, fresh_snapshot
from dual_runtime import scalebfm_residual_policy as runtime
from deploy_dual_scalebfm_residual import resolve_artifacts, build_parser


@pytest.fixture
def base_policy(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.npz'
    write_reference(reference)
    targets = np.stack([np.full(29, .2), np.full(29, -.3)]).astype(np.float32)
    actions = targets * 2
    base = SimpleNamespace(infer_batch=lambda *args, **kwargs: (targets.copy(), actions.copy()))
    monkeypatch.setattr(runtime, 'ScaleBFMPolicy', lambda *args, **kwargs: base)
    monkeypatch.setattr(runtime, 'G1PolicyKinematics', lambda *args: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('base mode must not load/invoke residual components')
    monkeypatch.setattr(runtime, 'ResidualPolicy', forbidden)
    monkeypatch.setattr(runtime, 'build_residual_observation', forbidden)
    policy = runtime.DualScaleBFMResidualPolicy(
        scalebfm_checkpoint='base', scalebfm_metadata='metadata', scalebfm_mode_table='modes',
        reference_bundle=reference, kinematics_xml='robot.xml', residual_enabled=False)
    live = SimpleNamespace(body_pos_w=np.zeros((len(policy.reference.body_order), 3)),
                           body_quat_wxyz=np.tile([1., 0, 0, 0], (len(policy.reference.body_order), 1)))
    policy._live_kinematics = lambda *args: (live, live)
    policy.initialized = True
    return policy, targets, actions


def test_base_targets_and_history_have_no_residual(base_policy):
    policy, expected, actions = base_policy
    assert policy.residual is None
    first = policy.compute((state(), state()), fresh_snapshot())
    np.testing.assert_array_equal(first.targets, expected)
    np.testing.assert_array_equal(first.residuals, 0)
    np.testing.assert_array_equal(policy.previous_executed_action, actions)
    second = policy.compute((state(), state()), fresh_snapshot())
    assert second.frame == first.frame + 1
    for index, history in enumerate(policy.histories):
        np.testing.assert_array_equal(history.arrays()[-1][-1], actions[index])
    assert not first.complete
    while not second.complete:
        second = policy.compute((state(), state()), fresh_snapshot())
    assert second.frame == policy.reference.frames - 1


def test_enabled_mode_still_adds_residual(base_policy, monkeypatch):
    policy, expected, actions = base_policy
    policy.residual_enabled = True
    policy.residual = SimpleNamespace(infer=lambda obs: np.full((2, 29), .5, dtype=np.float32))
    policy.scalebfm.default_q = np.zeros(29)
    policy.scalebfm.action_scale = np.full(29, .5)
    monkeypatch.setattr(runtime, 'build_residual_observation', lambda **kwargs: np.zeros(201))
    result = policy.compute((state(), state()), fresh_snapshot())
    np.testing.assert_allclose(result.targets, expected + .05)
    np.testing.assert_allclose(policy.previous_executed_action, actions + .1)


def test_base_artifact_validation_skips_actor_but_checks_base(tmp_path):
    names = ('scalebfm_checkpoint', 'scalebfm_metadata', 'scalebfm_mode_table',
             'reference_bundle', 'kinematics_xml')
    files = {}
    config = {'artifacts': {'directory': str(tmp_path), 'manifest': 'manifest.json'}}
    for name in names:
        p = tmp_path / name
        p.write_bytes(b'checked-resource')
        config['artifacts'][name] = name
        files[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'task': 'dual_g1_scalebfm_residual_object', 'files': files}))
    assert set(resolve_artifacts(tmp_path / 'config.yaml', config, include_residual=False)) == set(names)
    with pytest.raises(KeyError):
        resolve_artifacts(tmp_path / 'config.yaml', config)
    (tmp_path / 'scalebfm_checkpoint').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        resolve_artifacts(tmp_path / 'config.yaml', config, include_residual=False)


@pytest.mark.parametrize('source', ['sim', 'vive'])
def test_both_inputs_accept_same_baseline_flag(source):
    args = build_parser().parse_args(['--pose-source', source, '--scalebfm-only'])
    assert args.scalebfm_only and args.pose_source == source


def test_no_object_tracker_required_for_empty_handed_vive(tmp_path):
    from dual_runtime.vive_dual_pose import DualViveDeploymentConfig, DualVivePoseProvider
    from test_dual_scalebfm_deploy import sample
    identity = {'position_m':[0,0,0], 'quaternion_xyzw':[0,0,0,1]}
    path = tmp_path/'two_trackers.json'
    path.write_text(json.dumps({'robot_a_tracker_serial':'A','robot_b_tracker_serial':'B',
        'world_from_steamvr':identity,'robot_a_tracker_to_pelvis':identity,
        'robot_b_tracker_to_pelvis':identity,'calibration_confirmed':True}))
    config = DualViveDeploymentConfig.load(path, require_object=False)
    requested = []
    def read(serials):
        requested.append(tuple(serials))
        return {'A':sample(0),'B':sample(1)}
    provider = DualVivePoseProvider(config, require_object=False,
                                  reader=SimpleNamespace(read_all=read))
    assert provider.update_once()
    assert requested == [('A','B')]
    assert provider.get_snapshot() is not None
    with pytest.raises(ValueError):
        DualViveDeploymentConfig.load(path)


def test_empty_handed_preflight_ignores_object_but_checks_partner(base_policy):
    from dataclasses import replace
    from omnicontact.contracts import RobotPose, ObjectPose
    from dual_runtime.dual_pose_provider import DualPoseSnapshot
    policy, _, _ = base_policy
    policy.initialized = False
    ref_a, ref_b = policy.reference.frame(1)
    snap = DualPoseSnapshot(RobotPose(ref_a.body_pos_w[0], np.array([0.,0.,0.,1.]),0),
        RobotPose(ref_b.body_pos_w[0],np.array([0.,0.,0.,1.]),0),
        ObjectPose(np.array([20.,20.,20.]),np.array([1.,0.,0.,0.]),np.ones(3)*5,0))
    limits=dict(max_partner_position_error_m=.2,max_object_position_error_m=.2,
        max_box_size_error_m=.03,max_robot_orientation_error_rad=.35,max_object_orientation_error_rad=.35)
    policy.initialize(snap, **limits)
    policy.initialized = False
    policy.reference = runtime.DualReferenceBundle(policy.reference.path)
    with pytest.raises(RuntimeError, match='robot B'):
        policy.initialize(replace(snap,robot_b=RobotPose(ref_b.body_pos_w[0]+10,np.array([0.,0.,0.,1.]),0)),**limits)


def test_world_preflight_checks_robot_a_without_moving_reference(base_policy):
    from omnicontact.contracts import RobotPose, ObjectPose
    from dual_runtime.dual_pose_provider import DualPoseSnapshot
    policy, _, _ = base_policy
    assert policy.reference_alignment == 'none'
    policy.initialized = False
    ref_a, ref_b = policy.reference.frame(policy.start_frame)
    a_before = ref_a.body_pos_w.copy()
    b_before = ref_b.body_pos_w.copy()
    snap = DualPoseSnapshot(
        RobotPose(ref_a.body_pos_w[0]+[1.,0.,0.], ref_a.body_quat_wxyz[0][[1,2,3,0]], 0),
        RobotPose(ref_b.body_pos_w[0], ref_b.body_quat_wxyz[0][[1,2,3,0]], 0),
        ObjectPose(np.zeros(3),np.array([0.,0.,0.,1.]),np.ones(3),0))
    with pytest.raises(RuntimeError, match='robot A initial position differs from world reference'):
        policy.initialize(snap,max_partner_position_error_m=.2,max_object_position_error_m=.2,
            max_box_size_error_m=.03,max_robot_orientation_error_rad=.35,max_object_orientation_error_rad=.35)
    after_a, after_b = policy.reference.frame(policy.start_frame)
    np.testing.assert_array_equal(after_a.body_pos_w, a_before)
    np.testing.assert_array_equal(after_b.body_pos_w, b_before)
