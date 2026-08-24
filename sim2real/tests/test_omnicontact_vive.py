import json
import socket
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from omnicontact.contracts import ObjectPose, RobotPose
from omnicontact.perception.openvr_tracker import ViveSample
from omnicontact.perception.pose_udp import (
    PoseUdpPublisher,
    UdpPoseReceiverProvider,
    decode_pose_packet,
    encode_pose_packet,
)
from omnicontact.perception.vive_pose import (
    RigidTransform,
    ViveDeploymentConfig,
    VivePoseProvider,
)


class FakeTrackerReader:
    def __init__(self, frames):
        self.frames = list(frames)
        self.serial_to_index = {"ROBOT": 1, "OBJECT": 2}

    def start(self):
        return dict(self.serial_to_index)

    def read_all(self, serials=None):
        del serials
        return self.frames.pop(0)

    def refresh_devices(self):
        return dict(self.serial_to_index)

    def stop(self):
        pass


def sample(position):
    return ViveSample(
        sec=1,
        nsec=2,
        line_x_m=position[0],
        line_y_m=position[1],
        line_z_m=position[2],
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )


def deployment_config():
    identity = RigidTransform([0, 0, 0], [0, 0, 0, 1])
    return ViveDeploymentConfig(
        robot_tracker_serial="ROBOT",
        object_tracker_serial="OBJECT",
        world_from_steamvr=identity,
        robot_tracker_to_pelvis=RigidTransform([0, 0, -0.1], [0, 0, 0, 1]),
        object_tracker_to_object=RigidTransform([0.1, 0, 0], [0, 0, 0, 1]),
        object_half_extents_m=[0.2, 0.3, 0.4],
        goal_position_w=[2.0, 0.0, 0.4],
        calibration_confirmed=True,
    )


class TestOmniContactVive(unittest.TestCase):
    def test_transform_composition_rotates_translation(self):
        parent = RigidTransform(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        )
        child = RigidTransform([1, 0, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(
            parent.compose(child).position,
            [1.0, 3.0, 3.0],
            atol=1e-6,
        )

    def test_config_rejects_non_boolean_or_unconfirmed_calibration(self):
        raw = {
            "calibration_confirmed": "false",
            "robot_tracker_serial": "ROBOT",
            "object_tracker_serial": "OBJECT",
            "world_from_steamvr": {
                "position_m": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            "robot_tracker_to_pelvis": {
                "position_m": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            "object_tracker_to_object": {
                "position_m": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            "object_half_extents_m": [0.1, 0.1, 0.1],
            "goal_position_w": [1, 0, 0.2],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vive.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration_confirmed"):
                ViveDeploymentConfig.load(path)

    def test_provider_publishes_atomic_pair_and_clears_on_loss(self):
        reader = FakeTrackerReader(
            [
                {"ROBOT": sample([1, 2, 3]), "OBJECT": sample([4, 5, 6])},
                {"ROBOT": sample([1, 2, 3]), "OBJECT": None},
            ]
        )
        provider = VivePoseProvider(deployment_config(), reader=reader)
        self.assertTrue(provider.update_once())
        robot, obj = provider.get_poses()
        np.testing.assert_allclose(robot.position_w, [1, 2, 2.9])
        np.testing.assert_allclose(obj.position_w, [4.1, 5, 6])
        tracker = provider.get_tracker_diagnostics(robot.stamp_s)
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker["tracker_sample_wall_time_ns"], 1_000_000_002)
        np.testing.assert_allclose(
            tracker["robot_tracker_position_w"], [1, 2, 3]
        )
        np.testing.assert_allclose(
            tracker["object_tracker_position_w"], [4, 5, 6]
        )
        self.assertFalse(provider.update_once())
        self.assertEqual(provider.get_poses(), (None, None))

    def test_authenticated_packet_round_trip_and_tamper_rejection(self):
        stamp = time.monotonic()
        robot = RobotPose([0, 0, 0.8], [0, 0, 0, 1], stamp)
        obj = ObjectPose(
            [1, 0, 0.3],
            [0, 0, 0, 1],
            [0.2, 0.2, 0.2],
            stamp,
        )
        packet = encode_pose_packet(
            robot,
            obj,
            sequence=7,
            stream_id="12345678-1234-5678-1234-567812345678",
            token="test-secret",
        )
        _, sequence, decoded_robot, decoded_object = decode_pose_packet(
            packet,
            "test-secret",
        )
        self.assertEqual(sequence, 7)
        np.testing.assert_allclose(decoded_robot.position_w, robot.position_w)
        np.testing.assert_allclose(decoded_object.position_w, obj.position_w)
        with self.assertRaisesRegex(ValueError, "authentication"):
            decode_pose_packet(packet, "wrong-secret")

    def test_udp_receiver_accepts_local_authenticated_pair(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        receiver = UdpPoseReceiverProvider(
            "127.0.0.1",
            port,
            "test-secret",
            allowed_sender_ip="127.0.0.1",
        )
        publisher = PoseUdpPublisher("127.0.0.1", port, "test-secret")
        stamp = time.monotonic()
        robot = RobotPose([0, 0, 0.8], [0, 0, 0, 1], stamp)
        obj = ObjectPose([1, 0, 0.3], [0, 0, 0, 1], [0.2, 0.2, 0.2], stamp)
        try:
            receiver.start()
            publisher.send(robot, obj)
            self.assertTrue(receiver.wait_until_ready(1.0))
            received_robot, received_object = receiver.get_poses()
            np.testing.assert_allclose(received_robot.position_w, robot.position_w)
            np.testing.assert_allclose(received_object.position_w, obj.position_w)
        finally:
            publisher.close()
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
