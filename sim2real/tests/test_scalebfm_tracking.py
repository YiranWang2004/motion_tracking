"""Single-agent contracts without DDS, SteamVR or actuation."""

import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zmq
from omnicontact.contracts import RobotPose
from omnicontact.runtime import BridgeState, CommandLimiter
from scalebfm.constants import POLICY_JOINT_NAMES
from scalebfm_tracking.config import load_config
from scalebfm_tracking.pose import LocalOdometry, ViveReader, ViveSource
from scalebfm_tracking.reference import Frame, PicoSource, ReferenceWindow, decode_frame
from scalebfm_tracking.runner import limit_command
from scipy.spatial.transform import Rotation


def state(q=None):
    return BridgeState(
        np.zeros(29) if q is None else q,
        np.zeros(29),
        np.array([1.0, 0, 0, 0]),
        np.zeros(3),
        {},
        None,
        1,
        0,
    )


class FK:
    def forward(self, joints, pose):
        # Both support feet shift relative to the pelvis when joint 0 changes.
        pos = np.zeros((14, 3))
        pos[[3, 6], 2] = -0.75
        pos[[3, 6], 0] = joints[0]
        pos = Rotation.from_quat(pose.quaternion_xyzw).apply(pos) + pose.position_w
        return SimpleNamespace(
            body_pos_w=pos,
            body_quat_wxyz=np.tile(pose.quaternion_xyzw[[3, 0, 1, 2]], (14, 1)),
        )


def test_decode_joint_order_and_reject_bad_contract():
    names = list(reversed(POLICY_JOINT_NAMES))
    header = {
        "protocol_version": 2,
        "joint_names": names,
        "num_frames": 1,
        "qpos_size": 36,
    }
    q = np.r_[1, 2, 3, 1, 0, 0, 0, np.arange(29)].astype(np.float32)
    frame = decode_frame(header, q.tobytes(), 5)
    np.testing.assert_array_equal(frame.joints, np.arange(29)[::-1])
    for bad in (
        {**header, "protocol_version": 1},
        {**header, "joint_names": names[:-1]},
        {**header, "qpos_size": 35},
    ):
        with pytest.raises(ValueError):
            decode_frame(bad, q.tobytes(), 5)
    q[4] = np.nan
    with pytest.raises(ValueError):
        decode_frame(header, q.tobytes(), 5)


def test_window_is_delayed_and_fixed_session_alignment():
    frames = [
        Frame(
            10 + i * 0.02,
            np.array([i * 0.02, 0, 1]),
            np.array([1, 0, 0, 0]),
            np.zeros(29),
        )
        for i in range(11)
    ]
    window = ReferenceWindow(FK(), transition_s=0.1)
    anchor = RobotPose(
        [4, 5, 0.8], Rotation.from_euler("z", 90, degrees=True).as_quat(), 10
    )
    window.reset(frames[0], anchor, np.zeros(29), 10)
    pos, quat = window.build(frames, 11)
    assert pos.shape == (1, 6, 14, 3) and quat.shape == (1, 6, 14, 4)
    # Future slot5 is latest known sample; slot0 is 100ms behind it.
    np.testing.assert_allclose(pos[0, 0, 0], [4, 5.1, 0.8], atol=1e-6)
    np.testing.assert_allclose(pos[0, 5, 0], [4, 5.2, 0.8], atol=1e-6)
    with pytest.raises(RuntimeError, match="warming"):
        window.build(frames[:3], 11)
    with pytest.raises(ValueError):
        ReferenceWindow(FK(), offsets=[0, 1, 2, 3, 5, 6])


def test_local_odometry_integrates_anchored_feet_and_removes_initial_yaw():
    odom = LocalOdometry(FK())
    s = replace(state(), quat_wxyz=np.array([2**-0.5, 0, 0, 2**-0.5]))
    p = odom.update(s, 1)
    np.testing.assert_allclose(p.position_w, [0, 0, 0.78], atol=1e-6)
    np.testing.assert_allclose(p.quaternion_xyzw, [0, 0, 0, 1], atol=1e-6)
    q = np.zeros(29)
    q[0] = -0.01
    p2 = odom.update(replace(s, q_lab=q), 1.02)
    np.testing.assert_allclose(p2.position_w, [0.01, 0, 0.78], atol=1e-6)
    assert p2.confidence < 1
    with pytest.raises(RuntimeError, match="gap"):
        odom.update(s, 2)


def test_local_loss_of_support_fails_closed():
    odom = LocalOdometry(FK())
    odom.update(state(), 1)
    odom.position[2] += 0.2
    with pytest.raises(RuntimeError, match="contact"):
        odom.update(state(), 1.02)


class Socket:
    def __init__(self, messages=()):
        self.messages = deque(messages)
        self.sent = []

    def recv_json(self, *args):
        if not self.messages:
            raise zmq.Again()
        return self.messages.popleft()

    def recv_multipart(self, *args):
        return self.recv_json()

    def send_json(self, value, *args):
        self.sent.append(value)


def pico():
    p = PicoSource.__new__(PicoSource)
    p.timeout = 0.25
    p.req, p.rep, p.ctrl = Socket(), Socket(), Socket()
    p.pending = {}
    p.counter = 0
    p.session_id = "test"
    p.enabled = False
    p.generation = 0
    p.buttons = None
    p.last_sender = p.origin_sender = p.origin_local = p.last_valid = None
    p.frames = deque(maxlen=150)
    return p


def reply(p, now, sender, age=0.01, start=False, stop=False):
    request_id = f"request-{now}"
    p.pending[request_id] = now - 0.01
    header = {
        "protocol_version": 2,
        "request_id": request_id,
        "joint_names": list(POLICY_JOINT_NAMES),
        "num_frames": 1,
        "qpos_size": 36,
        "sample_age_s": age,
        "sample_time_ns": sender,
        "control_age_s": 0.01,
        "controller_buttons": {"right_key_one": start, "left_key_one": stop},
    }
    q = np.r_[0, 0, 1, 1, 0, 0, 0, np.zeros(29)].astype(np.float32)
    p.rep.messages.append((json.dumps(header).encode(), q.tobytes()))
    p.poll(now)


def test_pico_repeated_sample_never_refreshes_and_buttons_need_release():
    p = pico()
    reply(p, 10, 1000000000, start=True)
    assert not p.enabled
    reply(p, 10.02, 1020000000)
    reply(p, 10.04, 1040000000, start=True)
    assert p.enabled and p.generation == 1
    last = p.last_valid
    reply(p, 10.06, 1040000000)
    assert p.last_valid == last
    reply(p, 10.08, 1080000000, age=5)
    assert p.last_valid == last
    reply(p, 10.1, 1100000000, stop=True)
    assert not p.enabled
    assert not p.fresh(11)


def test_vive_rtt_and_imu_alignment():
    v = ViveSource.__new__(ViveSource)
    v.socket = Socket(
        [{"id": 1, "valid": True, "p": [1, 2, 3], "q": [0, 0, 2**-0.5, 2**-0.5]}]
    )
    v.pending = {1: 10.0}
    v.counter = 1
    v.timeout = 0.1
    v.latest = v.alignment = None
    pose = v.update(state(), 10.02)
    np.testing.assert_allclose(pose.position_w, [1, 2, 3])
    np.testing.assert_allclose(
        pose.quaternion_xyzw, [0, 0, 2**-0.5, 2**-0.5], atol=1e-6
    )
    with pytest.raises(RuntimeError, match="stale"):
        v.update(state(), 10.2)


def test_vive_mount_transform_order():
    v = ViveReader.__new__(ViveReader)
    v.serial = "one"
    v.world = {"position_m": [1, 0, 0], "quaternion_xyzw": [0, 0, 2**-0.5, 2**-0.5]}
    v.mount = {"position_m": [1, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
    sample = SimpleNamespace(line_x_m=1, line_y_m=0, line_z_m=0, qx=0, qy=0, qz=0, qw=1)
    v.reader = SimpleNamespace(read_all=lambda serials: {"one": sample})
    np.testing.assert_allclose(v.read().position_w, [1, 2, 0], atol=1e-6)


def test_command_limits_intersect_and_history_can_use_executed_action():
    limiter = CommandLimiter(np.full(29, -1), np.ones(29), 0.1)
    limiter.reset(np.zeros(29))
    policy = SimpleNamespace(
        kp=np.full(29, 100), kd=np.ones(29), torque_limit=np.full(29, 5)
    )
    cmd = limit_command(np.ones(29), state(), policy, limiter)
    np.testing.assert_allclose(cmd.target_pos, 0.05)
    with pytest.raises(RuntimeError, match="no target"):
        limit_command(
            np.ones(29), replace(state(), q_lab=np.full(29, 10)), policy, limiter
        )


def test_config_requires_no_partner_residual_or_reference():
    root = Path(__file__).resolve().parents[1]
    cfg, files = load_config(root / "config/g1/tracking_scalebfm.yaml")
    assert set(files) == {"checkpoint", "metadata", "mode_table", "xml"}
    assert cfg["pose_mode"] == "local"


def test_teleop_buffer_preserves_actual_sample_age_on_fallback():
    import multiprocessing

    from teleop.utils.buffer import SharedRetargetFrameRingBuffer

    ring = SharedRetargetFrameRingBuffer(
        multiprocessing.get_context("spawn"), qpos_size=36, capacity=10
    )
    assert ring.sample(target_ns=100).qpos is None
    q = np.r_[0, 0, 1, 1, 0, 0, 0, np.zeros(29)]
    for stamp in (100, 200):
        ring.append(
            recv_ns=stamp,
            seq=stamp,
            qpos=q,
            human_positions=None,
            human_rotations_wxyz=None,
            dropped_before_process=0,
            window_ns=1000,
        )
    assert ring.sample(target_ns=150).info["sample_time_ns"] == 150
    assert ring.sample(target_ns=10000).info["sample_time_ns"] == 200
    assert ring.sample(target_ns=50).info["sample_time_ns"] == 200


def test_old_imports_are_same_classes():
    from dual_runtime.kinematics import G1PolicyKinematics as old_fk
    from dual_runtime.scalebfm_policy import ScaleBFMPolicy as old
    from scalebfm.kinematics import G1PolicyKinematics as new_fk
    from scalebfm.policy import ScaleBFMPolicy as new

    assert old is new and old_fk is new_fk


@pytest.mark.parametrize("actuate", [False, True])
def test_runner_observe_is_read_only_and_active_fault_damps(monkeypatch, actuate):
    import time

    from scalebfm.kinematics import G1PolicyKinematics
    from scalebfm_tracking import runner

    cfg, files = load_config(
        Path(__file__).resolve().parents[1] / "config/g1/tracking_scalebfm.yaml"
    )
    q = np.asarray(json.loads(files["metadata"].read_text())["default_dof_pos"])
    policy = SimpleNamespace(
        default_q=q,
        action_scale=np.ones(29),
        kp=np.ones(29),
        kd=np.ones(29),
        torque_limit=np.full(29, 100),
    )
    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            clients.append(self)
            self.sent = []
            self.latest_state = None
            self.button_rise = {"start": True, "A": False}
            self.calls = 0

        def read_next(self, timeout):
            self.calls += 1
            if self.calls > 2:
                return None
            self.latest_state = replace(
                state(q), buttons={"stop": False}, packet_arrival_ns=time.monotonic_ns()
            )
            return self.latest_state

        def send(self, command, **kwargs):
            self.sent.append("command")

        def send_damping(self, state):
            self.sent.append("damping")

        def close(self):
            self.closed = True

    class Source:
        def __init__(self, *args):
            pass

        def poll(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "MotionBridgeClient", Client)
    monkeypatch.setattr(runner, "PicoSource", Source)
    fk = G1PolicyKinematics(files["xml"])
    if actuate:
        with pytest.raises(RuntimeError, match="timeout"):
            runner.run(cfg, policy, fk, actuate=True, duration=0.15)
        assert clients[0].sent[-1] == "damping"
    else:
        runner.run(cfg, policy, fk, duration=0.1)
        assert clients[0].sent == []
        assert clients[0].bridge_session is None
    assert clients[0].closed


@pytest.mark.parametrize("fault", [None, "pico", "compute"])
def test_full_policy_lifecycle_and_discard_faulted_targets(monkeypatch, fault):
    import time

    from scalebfm.kinematics import G1PolicyKinematics
    from scalebfm_tracking import runner

    cfg, files = load_config(
        Path(__file__).resolve().parents[1] / "config/g1/tracking_scalebfm.yaml"
    )
    cfg["stand_s"] = 0.005
    q = np.asarray(json.loads(files["metadata"].read_text())["default_dof_pos"])
    recorded = []

    class Policy:
        default_q = q
        action_scale = np.ones(29)
        kp = kd = np.ones(29)
        torque_limit = np.full(29, 100)

        def infer_batch(self, histories, pos, quat, *args, **kwargs):
            recorded.append(pos.copy())
            assert pos.shape == (1, 6, 14, 3) and quat.shape == (1, 6, 14, 4)
            assert len(histories) == 1
            if fault == "compute":
                time.sleep(0.09)
            return q[None], np.zeros((1, 29))

    class Source:
        def __init__(self, *args):
            self.enabled = False
            self.generation = 0
            self.calls = 0

        def poll(self, now):
            self.calls += 1
            self.frames = [
                Frame(
                    now - 0.2 + i * 0.02,
                    np.array([i * 0.02, 0, 1.0]),
                    np.array([1, 0, 0, 0]),
                    q,
                )
                for i in range(11)
            ]
            if self.calls == 4:
                self.enabled = True
                self.generation = 1
            if self.calls == 6:
                self.enabled = False

        def fresh(self, now):
            return fault != "pico"

        def close(self):
            pass

    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.latest_state = None
            self.button_rise = {"start": True, "A": True}
            self.sent = []
            clients.append(self)

        def read_next(self, *args):
            self.calls += 1
            self.latest_state = replace(
                state(q),
                buttons={"stop": self.calls >= 8},
                packet_arrival_ns=time.monotonic_ns(),
            )
            return self.latest_state

        def send(self, command, **kwargs):
            self.sent.append(self.calls)

        def send_damping(self, state):
            self.sent.append("damping")

        def close(self):
            pass

    monkeypatch.setattr(runner, "MotionBridgeClient", Client)
    monkeypatch.setattr(runner, "PicoSource", Source)
    if fault:
        with pytest.raises(
            RuntimeError, match="stale" if fault == "pico" else "deadline"
        ):
            runner.run(
                cfg,
                Policy(),
                G1PolicyKinematics(files["xml"]),
                actuate=True,
                duration=0.5,
            )
        # The faulted cycle must not publish its target.
        assert clients[0].calls not in clients[0].sent
    else:
        runner.run(
            cfg, Policy(), G1PolicyKinematics(files["xml"]), actuate=True, duration=0.5
        )
        assert len(recorded) >= 4
        np.testing.assert_allclose(
            recorded[-1], np.repeat(recorded[-1][:, :1], 6, axis=1)
        )
    assert clients[0].sent[-1] == "damping"
