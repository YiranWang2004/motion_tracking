import json
import socket
import time
from pathlib import Path

import mujoco
import numpy as np

from dual_runtime.constants import KEY_BODY_NAMES, POLICY_JOINT_NAMES
from dual_runtime.visualization import (
    PROTOCOL,
    DualVisualizationReceiver,
    DualVisualizationSender,
    decode_visualization,
    encode_visualization,
)
from dual_runtime.visualization_replay import DualScaleBFMReplay
from scripts.view_dual_scalebfm_residual import (
    apply_bridge_joint_state,
    apply_visualization,
    decode_bridge_joint_state,
    load_twin,
)


ROOT = Path(__file__).resolve().parents[1]


def visualization_packet():
    identity = np.array([1, 0, 0, 0], dtype=np.float32)
    actual_bases = np.zeros((2, 7), dtype=np.float32)
    actual_bases[:, 3:] = identity
    actual_bases[0, :3] = [1, 2, 0.8]
    actual_bases[1, :3] = [1, 3, 0.8]
    reference_bases = actual_bases.copy()
    reference_bases[:, 0] += 0.1
    return encode_visualization(
        actual_robot_base_wxyz=actual_bases,
        actual_object_wxyz=[1, 2.5, 0.4, 1, 0, 0, 0],
        actual_box_half_extents=[0.3, 0.15, 0.12],
        reference_robot_base_wxyz=reference_bases,
        reference_joint_pos=np.full((2, 29), 0.2),
        reference_object_wxyz=[1.1, 2.5, 0.4, 1, 0, 0, 0],
        reference_box_half_extents=[0.32, 0.14, 0.11],
        target_joint_pos=np.full((2, 29), -0.3),
        deployment_state="executing",
        frame=7,
        scalebfm_target=np.zeros((2, 29)),
        residual=np.ones((2, 29)),
    )


def test_visualization_protocol_round_trip_and_rejects_wrong_shape():
    packet = visualization_packet()
    decoded = decode_visualization(packet)
    assert decoded is not None
    assert decoded["protocol"] == PROTOCOL
    assert decoded["policy"]["frame"] == 7
    np.testing.assert_allclose(
        decoded["actual"]["robot_base_wxyz"],
        packet["actual"]["robot_base_wxyz"],
    )
    malformed = dict(packet)
    malformed["actual"] = dict(packet["actual"])
    malformed["actual"]["robot_base_wxyz"] = np.zeros((1, 7))
    assert decode_visualization(malformed) is None


def test_visualization_udp_round_trip_uses_dedicated_read_only_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    receiver = DualVisualizationReceiver("127.0.0.1", port)
    sender = DualVisualizationSender("127.0.0.1", port)
    receiver.start()
    try:
        sender.send(visualization_packet())
        deadline = time.monotonic() + 1.0
        decoded = None
        while decoded is None and time.monotonic() < deadline:
            decoded = receiver.read_latest()
            time.sleep(0.005)
        assert decoded is not None
        assert decoded["policy"]["deployment_state"] == "executing"
    finally:
        sender.close()
        receiver.close()


def test_dual_twin_loads_and_applies_independent_a_b_state():
    model, data, bindings = load_twin(
        ROOT / "config/g1/assets/dual_scalebfm_twin.xml"
    )
    assert model.nmocap == 3
    assert len(set(bindings.actual_joint_qpos.reshape(-1).tolist())) == 58
    packet = visualization_packet()
    apply_visualization(model, data, bindings, packet, ghost_mode="reference")
    q_a = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    q_b = np.linspace(0.3, -0.3, 29, dtype=np.float32)
    apply_bridge_joint_state(data, bindings, 0, q_a)
    apply_bridge_joint_state(data, bindings, 1, q_b)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos[bindings.actual_joint_qpos[0]], q_a)
    np.testing.assert_allclose(data.qpos[bindings.actual_joint_qpos[1]], q_b)
    np.testing.assert_allclose(
        data.qpos[bindings.reference_joint_qpos], 0.2, atol=1.0e-7
    )
    np.testing.assert_allclose(
        data.qpos[bindings.actual_base_qpos[0] : bindings.actual_base_qpos[0] + 3],
        [1, 2, 0.8],
    )
    np.testing.assert_allclose(
        model.geom_size[bindings.actual_box_geom, :3], [0.3, 0.15, 0.12]
    )
    apply_visualization(model, data, bindings, packet, ghost_mode="target")
    np.testing.assert_allclose(data.qpos[bindings.reference_joint_qpos], -0.3)
    np.testing.assert_allclose(
        data.qpos[
            bindings.reference_base_qpos[1] : bindings.reference_base_qpos[1] + 3
        ],
        [1, 3, 0.8],
    )


def test_bridge_state_decoder_is_fail_closed():
    q = np.arange(29, dtype=np.float32)
    np.testing.assert_array_equal(decode_bridge_joint_state({"q": q}), q)
    assert decode_bridge_joint_state({"q": np.zeros(28)}) is None
    q[3] = np.nan
    assert decode_bridge_joint_state({"q": q}) is None
    assert decode_bridge_joint_state({"dq": np.zeros(29)}) is None


def _write_reference(path: Path) -> None:
    frames = 8
    body_count = len(KEY_BODY_NAMES)
    positions_a = np.zeros((frames, body_count, 3), dtype=np.float32)
    positions_a[..., 2] = 0.8
    positions_b = positions_a.copy()
    positions_b[..., 1] = 1.0
    quaternions = np.zeros((frames, body_count, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    joint = np.zeros((frames, 29), dtype=np.float32)
    body_velocity = np.zeros((frames, body_count, 3), dtype=np.float32)
    fields = {
        "fps": np.float32(50),
        "training_robot_count": np.int32(2),
        "training_joint_order": np.asarray(POLICY_JOINT_NAMES),
        "training_body_order": np.asarray(KEY_BODY_NAMES),
        "training_object_body_pos_w": np.tile([0, 0.5, 0.4], (frames, 1)),
        "training_object_body_quat_w": np.tile([1, 0, 0, 0], (frames, 1)),
        "training_box_half_extents": np.asarray([0.3, 0.15, 0.12]),
    }
    for index, positions in enumerate((positions_a, positions_b)):
        fields[f"training_robot_{index}_joint_pos"] = joint
        fields[f"training_robot_{index}_joint_vel"] = joint
        fields[f"training_robot_{index}_body_pos_w"] = positions
        fields[f"training_robot_{index}_body_quat_w"] = quaternions
        fields[f"training_robot_{index}_body_ang_vel_w"] = body_velocity
    np.savez(path, **fields)


def test_rollout_replay_reconstructs_reference_and_actual_state(tmp_path):
    reference = tmp_path / "reference.npz"
    _write_reference(reference)
    frames = 3
    robot_position = np.zeros((frames, 2, 3), dtype=np.float32)
    robot_position[:, :, 2] = 0.8
    robot_position[:, 1, 1] = 1.0
    robot_quaternion = np.zeros((frames, 2, 4), dtype=np.float32)
    robot_quaternion[..., 3] = 1.0
    object_position = np.tile([0, 0.5, 0.4], (frames, 1)).astype(np.float32)
    object_quaternion = np.zeros((frames, 4), dtype=np.float32)
    object_quaternion[:, 3] = 1.0
    np.savez(
        tmp_path / "rollout.npz",
        time_ns=np.asarray([10, 20_000_010, 40_000_010], dtype=np.int64),
        deployment_state=np.asarray(["default_pose", "executing", "executing"]),
        q=np.zeros((frames, 2, 29), dtype=np.float32),
        dq=np.zeros((frames, 2, 29), dtype=np.float32),
        imu_quat_wxyz=np.tile([1, 0, 0, 0], (frames, 2, 1)),
        gyro=np.zeros((frames, 2, 3), dtype=np.float32),
        robot_position_w=robot_position,
        robot_quat_xyzw=robot_quaternion,
        object_position_w=object_position,
        object_quat_xyzw=object_quaternion,
        object_half_extents=np.tile([0.31, 0.16, 0.13], (frames, 1)),
        command_target=np.full((frames, 2, 29), 0.25, dtype=np.float32),
        frame=np.asarray([-1, 1, 2]),
        scalebfm_target=np.zeros((frames, 2, 29), dtype=np.float32),
        residual=np.zeros((frames, 2, 29), dtype=np.float32),
        observation=np.zeros((frames, 2, 201), dtype=np.float32),
        inference_time_s=np.zeros(frames, dtype=np.float32),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "reference": str(reference),
                "reference_alignment": "xyyaw",
                "start_frame": 1,
            }
        ),
        encoding="utf-8",
    )
    replay = DualScaleBFMReplay(tmp_path / "rollout.npz")
    assert replay.frame_count == 3
    np.testing.assert_allclose(replay.elapsed_s, [0.0, 0.02, 0.04])
    packet = replay.visualization_packet(2)
    assert packet["policy"]["frame"] == 2
    np.testing.assert_allclose(packet["policy"]["target_joint_pos"], 0.25)
    np.testing.assert_allclose(
        packet["actual"]["box_half_extents"], [0.31, 0.16, 0.13]
    )
    np.testing.assert_allclose(
        packet["reference"]["robot_base_wxyz"][1, :3], [0, 1, 0.8]
    )
