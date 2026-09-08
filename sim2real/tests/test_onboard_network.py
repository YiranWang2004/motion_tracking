import socket
import time

import numpy as np
import pytest

from dual_runtime.onboard_network import ClockEstimate, LatestChannel, decode_pose
from dual_runtime.onboard_team import TeamState


def test_clock_offset_uses_four_timestamps_and_bounds_delay():
    clock = ClockEstimate(.02)
    clock.observe(100., 105.004, 105.005, 100.009, 1.)
    offset, uncertainty = clock.estimate(1.)
    assert offset == pytest.approx(5.)
    assert uncertainty == pytest.approx(.004)
    clock.observe(100., 106., 106., 102., 1.1)  # excessive RTT ignored
    assert clock.estimate(1.2)[0] == pytest.approx(5.)
    with pytest.raises(RuntimeError, match="expired"):
        clock.estimate(3.1)


def payload(captured=105.):
    return dict(calibration_id="test", snapshot_seq=1, captured=captured,
                poses=[[1, 2, 3, 0, 0, 0, 1], [2, 3, 4, 0, 0, 0, 1],
                       [0, 0, .15, 0, 0, 0, 1]], half_extents=[.5, .15, .15])


def test_pose_age_tracks_capture_not_reception_and_preserves_coordinates():
    kwargs = dict(calibration_id="test", offset=5., uncertainty=.004, max_age_s=.1,
                  wall=100.02, mono=20.)
    snapshot = decode_pose(payload(), **kwargs)
    np.testing.assert_allclose(snapshot.robot_a.position_w, [1, 2, 3])
    assert snapshot.robot_a.stamp_s == pytest.approx(19.976)
    with pytest.raises(RuntimeError, match="capture"):
        decode_pose(payload(104.), **kwargs)
    with pytest.raises(RuntimeError, match="capture"):
        decode_pose(payload(106.), **kwargs)
    with pytest.raises(RuntimeError, match="calibration"):
        decode_pose({**payload(), "calibration_id": "other"}, **kwargs)
    bad = payload()
    bad["poses"][0][3:] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="quaternion"):
        decode_pose(bad, **kwargs)


def port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(fn):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        try:
            return fn()
        except RuntimeError:
            time.sleep(.01)
    return fn()


def test_real_udp_clock_handshake_and_restart_fault():
    a_port, b_port = port(), port()
    a = LatestChannel(("127.0.0.1", a_port), ("127.0.0.1", b_port), "pose")
    b = LatestChannel(("127.0.0.1", b_port), ("127.0.0.1", a_port), "pose")
    try:
        b.publish(payload())
        received, offset, _ = wait_for(lambda: a.read(1.))
        assert received["snapshot_seq"] == 1
        assert abs(offset) < .01
        seq = a.remote_seq
        b.seq = -1
        b.publish(dict(snapshot_seq=99))  # replayed sequence cannot replace input
        time.sleep(.03)
        assert a.remote_seq == seq
        assert a.read(1.)[0]["snapshot_seq"] == 1
        b.stream = "restarted"
        b.publish(payload())
        time.sleep(.03)
        with pytest.raises(RuntimeError, match="restarted"):
            a.read(1.)
    finally:
        a.close()
        b.close()


def simulate_transition(a, b, event, now):
    b.request(event)
    changed = []
    for i in range(60):
        t = now + .02*i
        sa, sb = a.status(ready=True, frame=0), b.status(ready=True, frame=0)
        for machine, peer in ((a, sb), (b, sa)):
            if machine.update(peer, t, ready=True):
                changed.append(machine.robot_id)
        if len(changed) == 2:
            return t
    raise AssertionError("transition did not complete")


def test_team_handshake_accepts_either_remote_without_per_tick_lockstep():
    a, b = TeamState("a", "same"), TeamState("b", "same")
    now = 100.
    for event, phase in (("start", "default"), ("standing", "standing"), ("task", "executing")):
        now = simulate_transition(a, b, event, now) + .1
        assert a.phase == b.phase == phase
        assert a.started_at == b.started_at
    for i in range(60):
        sa, sb = a.status(ready=True, frame=10), b.status(ready=True, frame=10)
        a.update(sb, now+i*.02, ready=True, complete=True)
        b.update(sa, now+i*.02, ready=True, complete=True)
    assert a.phase == b.phase == "finished"


def test_lost_commit_ack_prevents_leader_activation():
    a, b = TeamState("a", "same"), TeamState("b", "same")
    a.request("start")
    a.update(b.status(ready=True, frame=-1), 100., ready=True)
    b.update(a.status(ready=True, frame=-1), 100.02, ready=True)
    a.update(b.status(ready=True, frame=-1), 100.04, ready=True)
    # Commit never reaches B. A must not activate merely because prepare was acked.
    with pytest.raises(RuntimeError, match="not_acknowledged"):
        a.update(b.status(ready=True, frame=-1), 100.76, ready=True)
    assert a.phase == "zero"


def test_team_rejects_mismatch_stop_and_old_button_events():
    a, b = TeamState("a", "same"), TeamState("b", "wrong")
    with pytest.raises(RuntimeError, match="configuration"):
        a.update(b.status(ready=True, frame=-1), 100., ready=True)
    b.fingerprint = "same"
    b.request("start")
    a.update(b.status(ready=False, frame=-1), 100., ready=True)
    a.update(b.status(ready=True, frame=-1), 100.1, ready=True)
    assert a.proposal is None  # no delayed activation from a pre-ready press
    b.request("stop")
    with pytest.raises(RuntimeError, match="operator_stop"):
        a.update(b.status(ready=True, frame=-1), 100.2, ready=True)
