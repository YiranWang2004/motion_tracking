import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from omnicontact.contracts import ObjectPose, PDCommand, RobotPose, TaskGoal
from omnicontact.loco_mode import LocoModePolicy
from omnicontact.policy import OmniContactCarryPolicy, RobotPolicyState
from omnicontact.runtime import (
    BridgePoseProvider,
    BridgeState,
    CommandLimiter,
    MotionBridgeClient,
    pose_pair_is_valid,
)
from omnicontact.visualization_udp import (
    VisualizationReceiver,
    VisualizationSender,
    decode_visualization,
    encode_visualization,
)
from deploy_omnicontact import _run_actuated
from paths import SIM2REAL_ROOT
from scripts import view_calibrated_omnicontact_poses as twin_viewer
from sim2sim import _set_pre_control_marker_geometry


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
        cls.loco_mode = LocoModePolicy(
            SIM2REAL_ROOT / "config/g1/omnicontact",
            controller["policy_joint_names"],
        )

    def setUp(self):
        self.policy.reset()
        self.loco_mode.reset()

    def test_locomode_observation_and_recurrent_reset(self):
        state = BridgeState(
            q_lab=self.loco_mode.default_lab.copy(),
            dq_lab=np.zeros(29, dtype=np.float32),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            buttons={},
            state_receive_time_ns=123,
            packet_seq=1,
            packet_arrival_ns=456,
        )
        observation = self.loco_mode.build_observation(state)
        self.assertEqual(observation.shape, (96,))
        np.testing.assert_allclose(observation[:3], 0.0)
        np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(observation[6:], 0.0)

        first = self.loco_mode.compute(state)
        second = self.loco_mode.compute(state)
        self.assertEqual(first.target_pos.shape, (29,))
        self.assertTrue(np.all(np.isfinite(first.target_pos)))
        self.assertFalse(np.allclose(first.target_pos, second.target_pos))
        self.loco_mode.reset()
        repeated_first = self.loco_mode.compute(state)
        np.testing.assert_allclose(repeated_first.target_pos, first.target_pos, atol=1e-6)

    def test_completed_cfgen_switches_to_locomode(self):
        def bridge_state(sequence: int, *, stop: bool = False) -> BridgeState:
            return BridgeState(
                q_lab=np.zeros(29, dtype=np.float32),
                dq_lab=np.zeros(29, dtype=np.float32),
                quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                gyro=np.zeros(3, dtype=np.float32),
                buttons={"stop": stop},
                state_receive_time_ns=sequence,
                packet_seq=sequence,
                packet_arrival_ns=sequence,
            )

        class FakeClient:
            def __init__(self):
                self.states = iter((bridge_state(2), bridge_state(3), bridge_state(4, stop=True)))
                self.button_rise = {}
                self.skipped_packets = 0
                self.commands = []

            def read_next(self, _timeout):
                state = next(self.states)
                self.button_rise = {"stop": state.buttons["stop"]}
                return state

            def send(self, command, **_kwargs):
                self.commands.append(command)

            def send_damping(self, _state):
                raise AssertionError("unexpected damping")

        class FakeTrackingPolicy:
            kp_lab = np.ones(29, dtype=np.float32)
            kd_lab = np.ones(29, dtype=np.float32)

            def __init__(self):
                self.advance_count = 0

            def compute(self, *_args):
                return SimpleNamespace(
                    command=PDCommand(
                        np.zeros(29, dtype=np.float32), self.kp_lab, self.kd_lab
                    ),
                    visualization=None,
                    task_state="trajectory_complete",
                )

            def advance(self):
                self.advance_count += 1

            def should_replan(self, *_args):
                raise AssertionError("completed trajectory must not replan")

        class FakeLocoMode:
            def __init__(self):
                self.reset_count = 0
                self.compute_count = 0

            def reset(self):
                self.reset_count += 1

            def compute(self, _state):
                self.compute_count += 1
                return PDCommand(
                    np.full(29, 0.1, dtype=np.float32),
                    np.ones(29, dtype=np.float32),
                    np.ones(29, dtype=np.float32),
                )

        stamp = time.monotonic()
        provider = SimpleNamespace(
            error=None,
            get_poses=lambda: (
                RobotPose([0, 0, 0.793], [0, 0, 0, 1], stamp),
                ObjectPose([1, 0, 0.15], [0, 0, 0, 1], [0.15, 0.15, 0.15], stamp),
            ),
        )
        client = FakeClient()
        tracking = FakeTrackingPolicy()
        loco_mode = FakeLocoMode()
        limiter = CommandLimiter(
            np.full(29, -10.0, dtype=np.float32),
            np.full(29, 10.0, dtype=np.float32),
            1.0,
        )
        final_state = _run_actuated(
            client,
            bridge_state(1),
            provider,
            tracking,
            loco_mode,
            TaskGoal([1, 1, 0.15]),
            limiter,
            pose_max_age_s=1.0,
            min_pose_confidence=0.9,
            state_timeout_s=1.0,
            run_seconds=None,
        )
        self.assertEqual(final_state.packet_seq, 4)
        self.assertEqual(tracking.advance_count, 1)
        self.assertEqual(loco_mode.reset_count, 1)
        self.assertEqual(loco_mode.compute_count, 1)
        self.assertEqual(len(client.commands), 2)
        np.testing.assert_allclose(client.commands[1].target_pos, 0.1)

    def test_carrybox_sim_scene_is_bundled_and_loadable(self):
        config_path = SIM2REAL_ROOT / "config/g1/bridge_omnicontact.yaml"
        bridge_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        scene_path = (config_path.parent / bridge_config["xml_path"]).resolve()

        self.assertTrue(scene_path.is_relative_to(SIM2REAL_ROOT))
        self.assertTrue(scene_path.is_file())
        model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
        self.assertEqual(model.nu, 29)
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box"),
            0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "ghost_floating_base_joint",
            ),
            0,
        )

    def test_all_sim_bridges_start_grounded_with_pre_control_marker(self):
        for relative_config in (
            "config/g1/bridge.yaml",
            "config/g1/bridge_omnicontact.yaml",
            "config/l7/bridge.yaml",
        ):
            with self.subTest(config=relative_config):
                config_path = SIM2REAL_ROOT / relative_config
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertNotIn("root_qpos_home", config)
                root_pose = np.asarray(config["root_qpos_control"], dtype=np.float64)
                self.assertLess(root_pose[2], 1.1)

                model_path = (config_path.parent / config["xml_path"]).resolve()
                model = mujoco.MjModel.from_xml_path(model_path.as_posix())
                data = mujoco.MjData(model)
                free_joint_ids = np.flatnonzero(
                    model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
                )
                self.assertGreater(len(free_joint_ids), 0)
                root_address = int(model.jnt_qposadr[int(free_joint_ids[0])])
                data.qpos[root_address : root_address + 7] = root_pose
                mujoco.mj_forward(model, data)

                marker = config["pre_control_marker"]
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, marker["body_name"]
                )
                self.assertGreaterEqual(body_id, 0)
                expected_position = data.xpos[body_id] + np.asarray(marker["offset"])
                scene = mujoco.MjvScene(model, maxgeom=2)
                _set_pre_control_marker_geometry(
                    scene,
                    visible=True,
                    position=expected_position,
                    radius=float(marker["radius"]),
                    half_length=float(marker["half_length"]),
                    rgba=np.asarray(marker["rgba"], dtype=np.float32),
                )
                self.assertEqual(scene.ngeom, 1)
                self.assertEqual(
                    scene.geoms[0].type, mujoco.mjtGeom.mjGEOM_CYLINDER
                )
                np.testing.assert_allclose(scene.geoms[0].pos, expected_position)
                np.testing.assert_allclose(scene.geoms[0].rgba, marker["rgba"])
                _set_pre_control_marker_geometry(
                    scene,
                    visible=False,
                    position=expected_position,
                    radius=float(marker["radius"]),
                    half_length=float(marker["half_length"]),
                    rgba=np.asarray(marker["rgba"], dtype=np.float32),
                )
                self.assertEqual(scene.ngeom, 0)

    def test_sim2sim_scene_has_world_pelvis_and_box_coordinate_axes(self):
        scene_path = SIM2REAL_ROOT / "config/g1/assets/omnicontact_carry_box.xml"
        model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
        expected_parent = {
            "world_frame": 0,
            "pelvis_frame": mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
            ),
            "box_frame": mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "box"
            ),
        }
        for prefix, body_id in expected_parent.items():
            self.assertGreaterEqual(body_id, 0)
            for suffix in ("origin", "axis_x", "axis_y", "axis_z"):
                geom_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{prefix}_{suffix}",
                )
                self.assertGreaterEqual(geom_id, 0)
                self.assertEqual(int(model.geom_bodyid[geom_id]), body_id)
                self.assertEqual(int(model.geom_contype[geom_id]), 0)
                self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)

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
        self.assertEqual(step.visualization.left_wrist_wxyz.shape, (7,))
        self.assertEqual(step.visualization.ghost_dof_pos.shape, (29,))

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

    def test_sim_command_contains_scene_and_reference_visualization(self):
        class FakeTransport:
            def __init__(self):
                self.kwargs = None

            def send_command(self, **kwargs):
                self.kwargs = kwargs
                return 7

        stamp = time.monotonic()
        robot = RobotPose([0, 0, 0.793], [0, 0, 0, 1], stamp)
        obj = ObjectPose([1, 0, 0.15], [0, 0, 0, 1], [0.15, 0.15, 0.15], stamp)
        goal = TaskGoal([1, 1, 0.15])
        self.policy.initialize_reference(robot, obj, goal)
        state = RobotPolicyState(
            self.policy.default_lab,
            np.zeros(29, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        step = self.policy.compute(state, robot, obj)

        client = MotionBridgeClient.__new__(MotionBridgeClient)
        client.transport = FakeTransport()
        client.pose_sink = BridgePoseProvider()
        client._carrybox_scene = None
        client.set_carrybox_scene(obj, goal)
        bridge_state = BridgeState(
            q_lab=self.policy.default_lab,
            dq_lab=np.zeros(29, dtype=np.float32),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            buttons={},
            state_receive_time_ns=123,
            packet_seq=1,
            packet_arrival_ns=456,
        )
        client.send(
            step.command,
            enable=1,
            state=bridge_state,
            visualization=step.visualization,
        )
        visual = client.transport.kwargs["extra_command"]["omnicontact_visualization"]
        np.testing.assert_allclose(
            visual["scene"]["start_plane_wxyz"],
            [1, 0, -0.01, 1, 0, 0, 0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            visual["scene"]["goal_plane_wxyz"],
            [1, 1, -0.01, 1, 0, 0, 0],
            atol=1e-7,
        )
        self.assertEqual(visual["reference"]["ghost_dof_pos"].shape, (29,))

    def test_read_only_visualization_packet_round_trip(self):
        scene = {
            "start_plane_wxyz": np.arange(7, dtype=np.float32),
            "goal_plane_wxyz": np.arange(7, dtype=np.float32) + 1,
        }
        reference = {
            "left_wrist_wxyz": np.zeros(7, dtype=np.float32),
            "right_wrist_wxyz": np.zeros(7, dtype=np.float32),
            "torso_wxyz": np.zeros(7, dtype=np.float32),
            "left_ankle_wxyz": np.zeros(7, dtype=np.float32),
            "right_ankle_wxyz": np.zeros(7, dtype=np.float32),
            "object_wxyz": np.zeros(7, dtype=np.float32),
            "contact": np.ones(4, dtype=np.float32),
            "ghost_base_wxyz": np.zeros(7, dtype=np.float32),
            "ghost_dof_pos": np.zeros(29, dtype=np.float32),
        }
        packet = encode_visualization(scene, reference)
        decoded = decode_visualization(packet)
        self.assertIsNotNone(decoded)
        np.testing.assert_allclose(
            decoded["scene"]["goal_plane_wxyz"], scene["goal_plane_wxyz"]
        )
        np.testing.assert_allclose(
            decoded["reference"]["ghost_dof_pos"], reference["ghost_dof_pos"]
        )

    def test_read_only_visualization_udp_sender_receiver(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
        probe.close()

        receiver = VisualizationReceiver("127.0.0.1", port)
        sender = VisualizationSender("127.0.0.1", port)
        receiver.start()
        try:
            sender.send(
                {
                    "start_plane_wxyz": np.arange(7, dtype=np.float32),
                    "goal_plane_wxyz": np.arange(7, dtype=np.float32) + 1,
                },
                None,
            )
            deadline = time.monotonic() + 1.0
            packet = None
            while packet is None and time.monotonic() < deadline:
                packet = receiver.read_latest()
                if packet is None:
                    time.sleep(0.01)
            self.assertIsNotNone(packet)
            np.testing.assert_allclose(
                packet["scene"]["goal_plane_wxyz"],
                np.arange(7, dtype=np.float32) + 1,
            )
        finally:
            sender.close()
            receiver.close()

    def test_read_only_visualization_does_not_require_sim_pose_sink(self):
        class FakeVisualizationSender:
            def __init__(self):
                self.calls = []

            def send(self, scene, reference):
                self.calls.append((scene, reference))

        stamp = time.monotonic()
        robot = RobotPose([0, 0, 0.793], [0, 0, 0, 1], stamp)
        obj = ObjectPose([1, 0, 0.15], [0, 0, 0, 1], [0.15, 0.15, 0.15], stamp)
        goal = TaskGoal([1, 1, 0.15])
        self.policy.initialize_reference(robot, obj, goal)
        state = RobotPolicyState(
            self.policy.default_lab,
            np.zeros(29, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        step = self.policy.compute(state, robot, obj)

        client = MotionBridgeClient.__new__(MotionBridgeClient)
        client.pose_sink = None
        client.visualization_sender = FakeVisualizationSender()
        client._carrybox_scene = None
        client.set_carrybox_scene(obj, goal)
        client.publish_visualization(step.visualization)

        self.assertEqual(len(client.visualization_sender.calls), 1)
        scene, reference = client.visualization_sender.calls[0]
        self.assertEqual(scene["start_plane_wxyz"].shape, (7,))
        self.assertEqual(reference["ghost_dof_pos"].shape, (29,))

    def test_sim2real_twin_scene_maps_complete_visualization(self):
        scene_path = SIM2REAL_ROOT / "config/g1/assets/omnicontact_carry_box.xml"
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".xml",
            dir=scene_path.parent,
            delete=False,
            encoding="utf-8",
        ) as stream:
            stream.write(twin_viewer._expanded_xml(scene_path))
            expanded_path = Path(stream.name)
        try:
            model = mujoco.MjModel.from_xml_path(expanded_path.as_posix())
        finally:
            expanded_path.unlink()
        data = mujoco.MjData(model)
        _, _, ghost_joint_qpos = twin_viewer._load_joint_setup(
            model, SIM2REAL_ROOT / "config/g1"
        )
        ghost_object_qpos = twin_viewer._qpos_addr(model, "ghost_box_joint")
        ghost_robot_qpos = twin_viewer._qpos_addr(
            model, "ghost_floating_base_joint"
        )
        body_names = {
            "start_plane_wxyz": "plane_1_holder",
            "goal_plane_wxyz": "plane_2_holder",
            "left_wrist_wxyz": "ref_l_wrist_frame",
            "right_wrist_wxyz": "ref_r_wrist_frame",
            "torso_wxyz": "ref_torso_frame",
            "left_ankle_wxyz": "ref_l_ankle_frame",
            "right_ankle_wxyz": "ref_r_ankle_frame",
        }
        visual_mocap_ids = {
            key: int(
                model.body_mocapid[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, value)
                ]
            )
            for key, value in body_names.items()
        }
        self.assertTrue(
            all(mocap_id >= 0 for mocap_id in visual_mocap_ids.values())
        )
        contact_geom_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in (
                    "ref_l_ankle_mesh",
                    "ref_r_ankle_mesh",
                    "ref_l_rubber_hand",
                    "ref_r_rubber_hand",
                )
            ],
            dtype=int,
        )
        reference_alpha = twin_viewer._reference_visual_alpha(model)
        twin_viewer._set_reference_visibility(model, reference_alpha, False)

        identity_pose = np.array([1, 2, 3, 1, 0, 0, 0], dtype=np.float32)
        visualization = {
            "scene": {
                "start_plane_wxyz": identity_pose,
                "goal_plane_wxyz": identity_pose + np.array(
                    [1, 0, 0, 0, 0, 0, 0], dtype=np.float32
                ),
            },
            "reference": {
                name: identity_pose.copy()
                for name in (
                    "left_wrist_wxyz",
                    "right_wrist_wxyz",
                    "torso_wxyz",
                    "left_ankle_wxyz",
                    "right_ankle_wxyz",
                    "object_wxyz",
                    "ghost_base_wxyz",
                )
            },
        }
        visualization["reference"]["contact"] = np.array(
            [0, 1, 0, 1], dtype=np.float32
        )
        visualization["reference"]["ghost_dof_pos"] = np.arange(
            29, dtype=np.float32
        )
        twin_viewer._apply_visualization(
            model,
            data,
            visualization,
            visual_mocap_ids,
            ghost_object_qpos,
            ghost_robot_qpos,
            ghost_joint_qpos,
            reference_alpha,
            contact_geom_ids,
        )

        np.testing.assert_allclose(
            data.qpos[ghost_object_qpos : ghost_object_qpos + 7], identity_pose
        )
        np.testing.assert_allclose(
            data.qpos[ghost_robot_qpos : ghost_robot_qpos + 7], identity_pose
        )
        np.testing.assert_allclose(data.qpos[ghost_joint_qpos], np.arange(29))
        self.assertTrue(
            all(model.geom_rgba[index, 3] > 0 for index in reference_alpha)
        )
        twin_viewer._set_reference_visibility(model, reference_alpha, False)
        self.assertTrue(
            all(model.geom_rgba[index, 3] == 0 for index in reference_alpha)
        )

    def test_tracker_pyramid_faces_point_outward(self):
        vertices = np.fromstring(
            twin_viewer._pyramid_vertices(), sep=" ", dtype=np.float64
        ).reshape(-1, 3)
        faces = np.fromstring(
            twin_viewer._pyramid_faces(), sep=" ", dtype=np.int64
        ).reshape(-1, 3)
        solid_center = np.mean(vertices, axis=0)
        self.assertEqual(faces.shape, (4, 3))
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]
            )
            outward = np.mean(triangle, axis=0) - solid_center
            self.assertGreater(float(np.dot(normal, outward)), 0.0)

    def test_tracker_to_pelvis_xyz_rpy_round_trip_handles_gimbal_lock(self):
        original = twin_viewer.RigidTransform(
            [0.01, -0.02, 0.07],
            [0.5, -0.5, 0.5, 0.5],
        )
        values = twin_viewer._transform_to_xyz_rpy(original)
        restored = twin_viewer._xyz_rpy_to_transform(values)

        np.testing.assert_allclose(restored.position, original.position)
        self.assertAlmostEqual(
            abs(float(np.dot(restored.quaternion_xyzw, original.quaternion_xyzw))),
            1.0,
            places=7,
        )

    def test_tracker_to_pelvis_save_only_updates_selected_transform(self):
        original_config = {
            "calibration_confirmed": True,
            "robot_tracker_serial": "robot",
            "object_tracker_serial": "object",
            "world_from_steamvr": {"keep": [1, 2, 3]},
            "robot_tracker_to_pelvis": {
                "position_m": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            "object_tracker_to_object": {"keep": "unchanged"},
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "omnicontact_vive.json"
            config_path.write_text(
                json.dumps(original_config), encoding="utf-8"
            )
            transform = twin_viewer.RigidTransform(
                [0.1, -0.2, 0.3],
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            )
            twin_viewer._save_robot_tracker_to_pelvis(config_path, transform)
            saved = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            saved["world_from_steamvr"], original_config["world_from_steamvr"]
        )
        self.assertEqual(
            saved["object_tracker_to_object"],
            original_config["object_tracker_to_object"],
        )
        np.testing.assert_allclose(
            saved["robot_tracker_to_pelvis"]["position_m"], transform.position
        )
        np.testing.assert_allclose(
            saved["robot_tracker_to_pelvis"]["quaternion_xyzw"],
            transform.quaternion_xyzw,
        )


if __name__ == "__main__":
    unittest.main()
