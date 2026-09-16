from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from dual_runtime.dual_pose_provider import DualPoseSnapshot
from dual_runtime.vive_recovery import ViveRecovery, recovery_settings, mix_snapshot
from dual_runtime.onboard_network import decode_pose, PoseUnavailable, PROTOCOL
from dual_runtime.onboard_team import TeamState
from dual_runtime.onboard_relay import allowed_packet
from omnicontact.contracts import RobotPose, ObjectPose


def snapshot(t, x=0.):
    return DualPoseSnapshot(RobotPose([x,0,.8],[0,0,0,1],t),
                            RobotPose([x,1,.8],[0,0,0,1],t),
                            ObjectPose([x,.5,.4],[0,0,0,1],[.5,.15,.15],t))


def pair():
    cfg = recovery_settings({'enabled': True})
    return [ViveRecovery(s, cfg) for s in 'ab']


def tick(machines, t, available=(True,True), x=0.):
    for m, valid in zip(machines, available):
        m.observe(snapshot(t,x) if valid else None, t, 1000+t)
    reports = [deepcopy(m.status()) for m in machines]
    for i,m in enumerate(machines):
        m.update(reports[1-i], 1000+t, 20)
    return [m.mode for m in machines]


def test_brief_one_sided_loss_recovers_at_shared_time():
    machines = pair()
    for t in np.arange(0., .2, .02):
        assert tick(machines, t) == ['normal','normal']
    for t in np.arange(.2,.6,.02):
        modes = tick(machines,t,(True,False))
    assert modes == ['grace','grace']
    resume_times = []
    observed = set()
    for t in np.arange(.6,1.8,.02):
        observed.update(tick(machines,t,x=.05))
        resume_times += [m.resumed_at for m in machines if m.resumed_at is not None]
    assert 'recovering' in observed
    assert [m.mode for m in machines] == ['normal','normal']
    assert len(resume_times) == 2
    assert resume_times[0] == resume_times[1]


def test_long_outage_latches_local_standing_and_never_auto_restarts():
    machines = pair()
    tick(machines,0.)
    for t in np.arange(.02,1.1,.02):
        tick(machines,t,(False,False))
        if t < 1.:
            assert all(m.mode != 'fallback' for m in machines)
    assert all(m.mode == 'fallback' for m in machines)
    for t in np.arange(1.1,2.,.02):
        assert tick(machines,t) == ['fallback','fallback']


def test_recovery_rejects_large_pose_jump():
    machines=pair()
    tick(machines,0.)
    for t in np.arange(.02,.3,.02):
        tick(machines,t,(False,False))
    assert tick(machines,.3,x=1.) == ['fallback','fallback']


def test_dropout_during_blend_cancels_resume():
    machines=pair()
    tick(machines,0.)
    for t in np.arange(.02,.3,.02):
        tick(machines,t,(False,False))
    for t in np.arange(.3,.7,.02):
        tick(machines,t)
    assert any(m.mode == 'recovering' for m in machines)
    for t in np.arange(.7,1.9,.02):
        tick(machines,t,(False,False))
    assert all(m.mode == 'fallback' for m in machines)


def test_stale_retransmissions_do_not_extend_one_second_deadline():
    machines=pair()
    tick(machines,0.)
    for t in np.arange(.02,1.1,.02):
        for m in machines:
            m.observe(snapshot(0.), t, 1000+t)
        reports=[deepcopy(m.status()) for m in machines]
        for i,m in enumerate(machines):
            m.update(reports[1-i],1000+t,20)
    assert all(m.mode=='fallback' for m in machines)


def test_new_packet_just_after_deadline_cannot_revive_task():
    machines=pair()
    tick(machines,0.)
    tick(machines,.99,(False,False))
    assert tick(machines,1.01) == ['fallback','fallback']


def test_duplicate_snapshot_sequence_never_counts_as_new_valid_sample():
    m=pair()[0]
    m.observe(snapshot(0.),0.,1000.,sample_seq=1)
    m.observe(snapshot(.02),.02,1000.02,sample_seq=1)
    assert m.good_samples == 1
    assert m.latest.robot_a.stamp_s == 0.


def test_missing_commit_ack_prevents_unilateral_resume():
    machines=pair()
    tick(machines,0.)
    for t in np.arange(.02,.3,.02):
        tick(machines,t,(False,False))
    for t in np.arange(.3,.45,.02):
        tick(machines,t)
    a,b=machines
    assert a.plan is not None
    t=a.plan['at']-1000
    a.observe(snapshot(t),t,1000+t)
    peer=deepcopy(b.status())
    peer['committed']=0
    with pytest.raises(RuntimeError,match='not_acknowledged'):
        a.update(peer,1000+t,20)


def test_slerp_short_arc_and_source_timestamp_preserved():
    a,b=snapshot(0),snapshot(1, .1)
    b=replace(b,robot_a=replace(b.robot_a, quaternion_xyzw=[0,0,0,-1]))
    c=mix_snapshot(a,b,.5)
    np.testing.assert_allclose(c.robot_a.position_w,[.05,0,.8])
    assert abs(c.robot_a.quaternion_xyzw[3]) == pytest.approx(1.)
    assert c.robot_a.stamp_s == 1.


def test_only_old_pose_is_recoverable_not_future_or_bad_calibration():
    payload=dict(calibration_id='c',captured=9., poses=[[0,0,0,0,0,0,1]]*3,
                 half_extents=[.5,.15,.15])
    args=dict(calibration_id='c',offset=0.,uncertainty=0.,max_age_s=.4,wall=10.,mono=1.)
    with pytest.raises(PoseUnavailable):
        decode_pose(payload,**args)
    for bad in ({**payload,'captured':11.},{**payload,'calibration_id':'bad'}):
        with pytest.raises(RuntimeError) as exc:
            decode_pose(bad,**args)
        assert not isinstance(exc.value,PoseUnavailable)


def test_relay_only_accepts_small_recovery_status_not_joint_payloads():
    machine=pair()[0]
    status=TeamState('a','f').status(ready=False,frame=1)
    status['vive_recovery']=machine.status()
    packet=dict(protocol=PROTOCOL,kind='team',stream='s',seq=1,sent=1.,payload=status)
    assert allowed_packet(json.dumps(packet).encode(),'team')
    status['vive_recovery']['target']=[0]*29
    assert not allowed_packet(json.dumps(packet).encode(),'team')
