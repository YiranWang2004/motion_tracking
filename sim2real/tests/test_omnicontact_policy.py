import time
import unittest
from pathlib import Path

import numpy as np
import yaml

from omnicontact.contracts import ObjectPose, PDCommand, RobotPose, TaskGoal
from omnicontact.policy import OmniContactCarryPolicy, RobotPolicyState
from omnicontact.runtime import (
    BridgePoseProvider,
    CommandLimiter,
    MotionBridgeClient,
    pose_pair_is_valid,
)
from paths import SIM2REAL_ROOT


class TestOmniContactPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller = yaml.safe_load(
            (SIM2REAL_ROOT / "config/g1/controller.yaml").read_text(encoding="utf-8")
        )
        cls.policy = OmniContactCarryPolicy(
            SIM2REAL_ROOT / "config/g1/omnicontact",
            controller["policy_joint_names"],
        )

    def setUp(self):
        self.policy.reset()

    def test_joint_permutations_are_inverse_and_match_controller_order(self):
        values = np.arange(29, dtype=np.float32)
        np.testing.assert_array_equal(
            self.policy.q_mj_to_lab(self.policy.q_lab_to_mj(values)),
            values,
        )

    def test_model_contract_is_two_inputs_and_1244_observations(self):
        self.assertEqual(self.policy.obs_input_name, "obs")
        self.assertEqual(self.policy.time_input_name, "time_step")

    def test_tracking_observation_places_contacts_after_pose_features(self):
        frames = 51
        zeros3 = np.zeros((frames, 3), dtype=np.float32)
        identity = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (frames, 1),
        )
        self.policy.reference = {
            "ref_left_wrist_pos": zeros3,
            "ref_left_wrist_quat": identity,
            "ref_right_wrist_pos": zeros3,
            "ref_right_wrist_quat": identity,
            "ref_torso_future_pos": zeros3,
            "ref_torso_future_quat": identity,
            "ref_left_ankle_future_pos": zeros3,
            "ref_left_ankle_future_quat": identity,
            "ref_right_ankle_future_pos": zeros3,
            "ref_right_ankle_future_quat": identity,
            "ref_contact": np.tile(
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                (frames, 1),
            ),
        }
        tracking = self.policy._tracking_observation(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(tracking.shape, (539,))
        np.testing.assert_allclose(
            tracking[495:],
            np.tile([1.0, 2.0, 3.0, 4.0], 11),
        )

    def test_reference_observation_and_onnx_smoke(self):
        stamp = time.monotonic()
        robot = RobotPose(
            [0.0, 0.0, 0.77],
            [0.0, 0.0, 0.0, 1.0],
            stamp,
        )
        obj = ObjectPose(
            [1.0, 0.0, 0.30],
            [0.0, 0.0, 0.0, 1.0],
            [0.15, 0.15, 0.15],
            stamp,
            linear_velocity_w=[0.0, 0.0, 0.0],
            angular_velocity_w=[0.0, 0.0, 0.0],
        )
        goal = TaskGoal([1.5, 0.0, 0.30])
        self.policy.initialize_reference(robot, obj, goal)
        state = RobotPolicyState(
            self.policy.default_lab,
            np.zeros(29, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        step = self.policy.compute(state, robot, obj)
        self.assertEqual(step.observation.shape, (1244,))
        self.assertEqual(step.command.target_pos.shape, (29,))
        self.assertTrue(np.all(np.isfinite(step.command.target_pos)))

    def test_completed_reference_keeps_policy_balance_command(self):
        stamp = time.monotonic()
        robot = RobotPose([0, 0, 0.793], [0, 0, 0, 1], stamp)
        obj = ObjectPose(
            [1, 0, 0.15],
            [0, 0, 0, 1],
            [0.15, 0.15, 0.15],
            stamp,
        )
        self.policy.initialize_reference(robot, obj, TaskGoal([1, 1, 0.15]))
        self.policy.frame = len(self.policy.reference["ref_contact"])
        state = RobotPolicyState(
            self.policy.default_lab,
            np.zeros(29, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        step = self.policy.compute(state, robot, obj)
        self.assertEqual(step.task_state, "trajectory_complete")
        self.assertEqual(step.observation.shape, (1244,))
        self.assertTrue(np.all(np.isfinite(step.command.target_pos)))
        self.assertFalse(self.policy.should_replan(obj, TaskGoal([3, 3, 0.15])))

    def test_command_limiter_clips_limits_and_delta(self):
        limiter = CommandLimiter(
            -np.ones(29, dtype=np.float32),
            np.ones(29, dtype=np.float32),
            0.1,
        )
        limiter.reset(np.zeros(29, dtype=np.float32))
        command = PDCommand(
            np.full(29, 5.0, dtype=np.float32),
            np.ones(29, dtype=np.float32),
            np.ones(29, dtype=np.float32),
        )
        safe = limiter.apply(command)
        np.testing.assert_allclose(safe.target_pos, 0.1, atol=1e-7)
        hold = limiter.hold(
            np.full(29, -0.5, dtype=np.float32),
            command.kp,
            command.kd,
        )
        np.testing.assert_allclose(hold.target_pos, -0.5)

    def test_pose_pair_freshness_requires_both_poses(self):
        stamp = time.monotonic()
        robot = RobotPose([0, 0, 0.77], [0, 0, 0, 1], stamp)
        obj = ObjectPose([1, 0, 0.3], [0, 0, 0, 1], [0.1, 0.1, 0.1], stamp)
        self.assertTrue(
            pose_pair_is_valid(
                robot,
                obj,
                max_age_s=0.1,
                min_confidence=0.9,
                now_s=stamp + 0.05,
            )
        )
        self.assertFalse(
            pose_pair_is_valid(
                robot,
                obj,
                max_age_s=0.1,
                min_confidence=0.9,
                now_s=stamp + 0.2,
            )
        )
        self.assertFalse(
            pose_pair_is_valid(
                robot,
                None,
                max_age_s=0.1,
                min_confidence=0.9,
                now_s=stamp,
            )
        )

    def test_sim_bridge_pose_is_published_as_one_pair(self):
        provider = BridgePoseProvider()
        client = MotionBridgeClient.__new__(MotionBridgeClient)
        client.pose_sink = provider
        client._publish_sim_pose(
            {
                "sim_pose": {
                    "robot": {
                        "position_w": [0, 0, 0.793],
                        "quaternion_xyzw": [0, 0, 0, 1],
                    },
                    "object": {
                        "position_w": [1, 0, 0.15],
                        "quaternion_xyzw": [0, 0, 0, 1],
                        "half_extents": [0.15, 0.15, 0.15],
                        "linear_velocity_w": [0, 0, 0],
                        "angular_velocity_w": [0, 0, 0],
                    },
                }
            }
        )
        robot, obj = provider.get_poses()
        np.testing.assert_allclose(robot.position_w, [0, 0, 0.793])
        np.testing.assert_allclose(obj.position_w, [1, 0, 0.15])
        self.assertEqual(robot.stamp_s, obj.stamp_s)


if __name__ == "__main__":
    unittest.main()
