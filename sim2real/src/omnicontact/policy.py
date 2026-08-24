"""Carry-box-only OmniContact policy adapter for motion_tracking.

The C++ bridge and the Python controller exchange joints in the policy/lab
order.  MuJoCo FK and CFgen use the grouped MuJoCo order, so this adapter owns
the only two explicit permutations between those spaces.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from omnicontact.contracts import (
    ObjectPose,
    PDCommand,
    ReferenceVisualization,
    RobotPose,
    TaskGoal,
)
from omnicontact.reference import CfGenCarryBox
from omnicontact.reference.loco_primitives import KINEMATICS
from omnicontact.reference.mujoco_kinematics_wxyz import MujocoKinematics
from omnicontact.reference.math_wxyz import (
    matrix_from_quat,
    quat_apply_batch,
    quat_conjugate,
    quat_mul_left_batch,
    quat_rotate_inverse,
    quat_to_6d_batch,
    subtract_frame_transforms,
    yaw_quat,
)


logger = logging.getLogger(__name__)


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    return quat[[3, 0, 1, 2]]


@dataclass(frozen=True)
class RobotPolicyState:
    """One LowState sample in motion_tracking's policy/lab joint order."""

    q_lab: np.ndarray
    dq_lab: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        q = np.asarray(self.q_lab, dtype=np.float32).reshape(-1)
        dq = np.asarray(self.dq_lab, dtype=np.float32).reshape(-1)
        angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32).reshape(3)
        if q.shape != (29,) or dq.shape != (29,):
            raise ValueError("OmniContact requires exactly 29 policy-order joints")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise ValueError("joint state contains non-finite values")
        if not np.all(np.isfinite(angular_velocity)):
            raise ValueError("angular velocity contains non-finite values")
        object.__setattr__(self, "q_lab", q.copy())
        object.__setattr__(self, "dq_lab", dq.copy())
        object.__setattr__(self, "angular_velocity", angular_velocity.copy())


@dataclass(frozen=True)
class PolicyStep:
    command: PDCommand
    observation: np.ndarray
    task_state: str
    visualization: ReferenceVisualization


class OmniContactCarryPolicy:
    """Exact 1244-D OmniContact tracker with carry-box CFgen references."""

    TRACKING_DIM_PER_FRAME = 49
    FUTURE_FRAMES = np.array([0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50], dtype=np.int32)
    HISTORY_SLICES = (
        (0, 15),
        (15, 18),
        (18, 21),
        (21, 50),
        (50, 79),
        (79, 108),
        (108, 111),
        (111, 117),
        (117, 141),
    )
    BBOX_SIGNS = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        asset_dir: str | Path,
        controller_joint_names: list[str],
        *,
        reference_pad: int = 30,
        replan_position_error: float = 1.50,
        replan_goal_error: float = 0.20,
        replan_linear_speed: float = 0.05,
        replan_angular_speed: float = 0.05,
        replan_cooldown_ticks: int = 80,
    ) -> None:
        self.asset_dir = Path(asset_dir).expanduser().resolve()
        with (self.asset_dir / "OmniContact.yaml").open("r", encoding="utf-8") as stream:
            model_cfg = yaml.safe_load(stream)

        self.session = ort.InferenceSession(
            (self.asset_dir / str(model_cfg["onnx_path"])).as_posix(),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise ValueError(f"OmniContact ONNX must have two inputs, got {len(inputs)}")
        obs_inputs = [item for item in inputs if item.shape and item.shape[-1] == 1244]
        if len(obs_inputs) != 1:
            raise ValueError("cannot identify the 1244-D OmniContact observation input")
        self.obs_input_name = obs_inputs[0].name
        self.time_input_name = next(item.name for item in inputs if item.name != self.obs_input_name)

        self.mj2lab = np.asarray(model_cfg["mj2lab"], dtype=np.int32)
        self.lab2mj = np.asarray(model_cfg["lab2mj"], dtype=np.int32)
        if (
            self.mj2lab.shape != (29,)
            or self.lab2mj.shape != (29,)
            or not np.array_equal(np.sort(self.mj2lab), np.arange(29))
            or not np.array_equal(np.sort(self.lab2mj), np.arange(29))
            or not np.array_equal(self.mj2lab[self.lab2mj], np.arange(29))
        ):
            raise ValueError("invalid OmniContact mj/lab joint permutations")

        expected_lab_names = [KINEMATICS.joint_names[index] for index in self.mj2lab]
        if list(controller_joint_names) != expected_lab_names:
            raise ValueError(
                "motion_tracking policy_joint_names do not match OmniContact lab order\n"
                f"expected={expected_lab_names}\nactual={list(controller_joint_names)}"
            )

        # CFgen uses its own module-level MuJoCo FK instance.  Keep policy FK
        # separate so an asynchronous replan cannot mutate the same MjData
        # while the control thread builds an observation (which can segfault
        # inside MuJoCo rather than raising a Python exception).
        self.kinematics = MujocoKinematics(
            (self.asset_dir / "g1_29dof_fk.xml").as_posix()
        )

        self.default_lab = np.asarray(model_cfg["default_angles_lab"], dtype=np.float32)
        self.action_scale_lab = np.asarray(model_cfg["action_scale_lab"], dtype=np.float32)
        self.kp_lab = np.asarray(model_cfg["kp_lab"], dtype=np.float32)
        self.kd_lab = np.asarray(model_cfg["kd_lab"], dtype=np.float32)
        self.lower_lab = np.asarray(model_cfg["joint_pos_lowerlimit_lab"], dtype=np.float32)
        self.upper_lab = np.asarray(model_cfg["joint_pos_upperlimit_lab"], dtype=np.float32)
        with (self.asset_dir / "DefaultPose.yaml").open("r", encoding="utf-8") as stream:
            default_pose_cfg = yaml.safe_load(stream)
        default_pose_mj = np.asarray(
            default_pose_cfg["default_angles"], dtype=np.float32
        )
        default_kp_mj = np.asarray(default_pose_cfg["kps"], dtype=np.float32)
        default_kd_mj = np.asarray(default_pose_cfg["kds"], dtype=np.float32)
        self.default_pose_lab = default_pose_mj[self.mj2lab]
        self.default_kp_lab = default_kp_mj[self.mj2lab]
        self.default_kd_lab = default_kd_mj[self.mj2lab]
        for name in (
            "default_lab",
            "action_scale_lab",
            "kp_lab",
            "kd_lab",
            "lower_lab",
            "upper_lab",
            "default_pose_lab",
            "default_kp_lab",
            "default_kd_lab",
        ):
            value = getattr(self, name)
            if value.shape != (29,) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid OmniContact parameter {name}")

        self.reference_pad = int(reference_pad)
        self.replan_position_error = float(replan_position_error)
        self.replan_goal_error = float(replan_goal_error)
        self.replan_linear_speed = float(replan_linear_speed)
        self.replan_angular_speed = float(replan_angular_speed)
        self.replan_cooldown_ticks = int(replan_cooldown_ticks)

        self.history = np.zeros((5, 141), dtype=np.float32)
        self.action_lab = np.zeros(29, dtype=np.float32)
        self.reference: dict[str, np.ndarray] | None = None
        self.frame = 0
        self.done = False
        self.replan_cooldown = 0
        self._reference_lock = threading.Lock()
        self._replan_thread: threading.Thread | None = None
        self._replan_pending: dict[str, np.ndarray] | None = None
        self._replan_error: BaseException | None = None
        self._next_replan_attempt_s = 0.0

    def q_lab_to_mj(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(29)
        return values[self.lab2mj]

    def q_mj_to_lab(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(29)
        return values[self.mj2lab]

    def reset(self) -> None:
        self.history.fill(0.0)
        self.action_lab.fill(0.0)
        self.reference = None
        self._replan_pending = None
        self._replan_error = None
        self._next_replan_attempt_s = 0.0
        self.frame = 0
        self.done = False
        self.replan_cooldown = 0

    def close(self) -> None:
        """Wait for a CFgen worker before its MuJoCo objects are torn down."""
        worker = self._replan_thread
        if worker is not None and worker.is_alive():
            worker.join()

    def _make_reference(
        self,
        robot_pose: RobotPose,
        object_pose: ObjectPose,
        goal: TaskGoal,
    ) -> dict[str, np.ndarray]:
        planner = CfGenCarryBox(pad=self.reference_pad)
        trajectory, _ = planner.generate(
            pelvis_pos=robot_pose.position_w,
            pelvis_quat=_xyzw_to_wxyz(robot_pose.quaternion_xyzw),
            obj_pos=object_pose.position_w,
            obj_quat=_xyzw_to_wxyz(object_pose.quaternion_xyzw),
            box_half_dims=object_pose.half_extents,
            target_obj_pos=goal.position_w,
        )
        return trajectory

    def initialize_reference(
        self,
        robot_pose: RobotPose,
        object_pose: ObjectPose,
        goal: TaskGoal,
    ) -> None:
        reference = self._make_reference(robot_pose, object_pose, goal)
        with self._reference_lock:
            self.reference = reference
            self._replan_pending = None
        self.frame = 0
        self.history.fill(0.0)
        self.action_lab.fill(0.0)
        self.done = False

    def _start_replan(
        self,
        robot_pose: RobotPose,
        object_pose: ObjectPose,
        goal: TaskGoal,
    ) -> None:
        if self._replan_thread is not None and self._replan_thread.is_alive():
            return

        def worker() -> None:
            try:
                result = self._make_reference(robot_pose, object_pose, goal)
                with self._reference_lock:
                    self._replan_pending = result
                    self._replan_error = None
            except BaseException as exc:
                with self._reference_lock:
                    self._replan_error = exc
                    self._next_replan_attempt_s = time.monotonic() + 1.0

        self._replan_thread = threading.Thread(
            target=worker,
            name="omnicontact-carry-replan",
            daemon=True,
        )
        self._replan_thread.start()

    def _apply_replan(self) -> bool:
        with self._reference_lock:
            if self._replan_pending is None:
                error = self._replan_error
                self._replan_error = None
                if error is not None:
                    logger.error("OmniContact replanning failed: %s", error)
                return False
            self.reference = self._replan_pending
            self._replan_pending = None
        self.frame = 0
        self.history.fill(0.0)
        self.action_lab.fill(0.0)
        self.done = False
        self.replan_cooldown = self.replan_cooldown_ticks
        logger.info("OmniContact carry-box reference replanned")
        return True

    def _fk(self, state: RobotPolicyState, robot_pose: RobotPose):
        return self.kinematics.forward(
            self.q_lab_to_mj(state.q_lab),
            robot_pose.position_w,
            _xyzw_to_wxyz(robot_pose.quaternion_xyzw),
        )

    def _tracking_observation(
        self,
        torso_pos: np.ndarray,
        torso_quat: np.ndarray,
    ) -> np.ndarray:
        assert self.reference is not None
        heading = yaw_quat(torso_quat).astype(np.float32)
        heading_conj = quat_conjugate(heading).astype(np.float32)
        last = len(self.reference["ref_left_wrist_pos"]) - 1
        indices = np.minimum(self.frame + self.FUTURE_FRAMES, last)

        def relative(pos_key: str, quat_key: str) -> tuple[np.ndarray, np.ndarray]:
            positions = self.reference[pos_key][indices]
            quaternions = self.reference[quat_key][indices]
            return (
                quat_apply_batch(heading_conj, positions - torso_pos[None, :]),
                quat_to_6d_batch(quat_mul_left_batch(heading_conj, quaternions)),
            )

        parts: list[np.ndarray] = []
        for pos_key, quat_key in (
            ("ref_left_wrist_pos", "ref_left_wrist_quat"),
            ("ref_right_wrist_pos", "ref_right_wrist_quat"),
            ("ref_torso_future_pos", "ref_torso_future_quat"),
            ("ref_left_ankle_future_pos", "ref_left_ankle_future_quat"),
            ("ref_right_ankle_future_pos", "ref_right_ankle_future_quat"),
        ):
            positions, rotations = relative(pos_key, quat_key)
            parts.extend((positions, rotations))
        tracking_pose = np.concatenate(parts, axis=-1).reshape(-1)
        tracking_contact = self.reference["ref_contact"][indices].reshape(-1)
        tracking = np.concatenate((tracking_pose, tracking_contact)).astype(np.float32)
        if tracking.shape != (539,):
            raise RuntimeError(f"OmniContact tracking observation has shape {tracking.shape}")
        return tracking

    def build_observation(
        self,
        state: RobotPolicyState,
        robot_pose: RobotPose,
        object_pose: ObjectPose,
    ) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("OmniContact reference is not initialized")
        fk = self._fk(state, robot_pose)
        torso_pos = fk["torso_link"]["pos"].astype(np.float32)
        torso_quat = fk["torso_link"]["quat"].astype(np.float32)
        heading = yaw_quat(torso_quat).astype(np.float32)
        ee_names = (
            "left_palm_link",
            "right_palm_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
            "mid360_link",
        )
        ee_pos = np.concatenate(
            [
                quat_rotate_inverse(torso_quat, fk[name]["pos"] - torso_pos)
                for name in ee_names
            ]
        ).astype(np.float32)
        object_quat = _xyzw_to_wxyz(object_pose.quaternion_xyzw)
        object_relative, object_rotation = subtract_frame_transforms(
            torso_pos,
            heading,
            object_pose.position_w,
            object_quat,
        )
        object_relative = self._clip_norm(object_relative)
        offsets = self.BBOX_SIGNS * object_pose.half_extents
        bbox_world = (
            quat_apply_batch(object_quat, offsets)
            + object_pose.position_w[None, :]
        )
        bbox_relative = self._clip_norm(
            quat_apply_batch(
                quat_conjugate(heading),
                bbox_world - torso_pos[None, :],
            )
        ).reshape(-1)
        current = np.concatenate(
            (
                ee_pos,
                state.angular_velocity,
                self._gravity(robot_pose.quaternion_xyzw),
                state.q_lab - self.default_lab,
                state.dq_lab,
                self.action_lab,
                object_relative,
                matrix_from_quat(object_rotation)[:, :2].reshape(-1),
                bbox_relative,
            )
        ).astype(np.float32)
        if current.shape != (141,):
            raise RuntimeError(f"OmniContact current observation has shape {current.shape}")
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = current
        flattened_history = np.concatenate(
            [self.history[:, start:end].reshape(-1) for start, end in self.HISTORY_SLICES]
        )
        observation = np.concatenate(
            (self._tracking_observation(torso_pos, torso_quat), flattened_history)
        ).astype(np.float32)
        if observation.shape != (1244,) or not np.all(np.isfinite(observation)):
            raise RuntimeError(
                f"invalid OmniContact observation shape/values: {observation.shape}"
            )
        return observation

    def compute(
        self,
        state: RobotPolicyState,
        robot_pose: RobotPose,
        object_pose: ObjectPose,
    ) -> PolicyStep:
        self._apply_replan()
        if self.reference is None:
            raise RuntimeError("initialize_reference must be called before compute")
        if self.frame >= len(self.reference["ref_contact"]):
            self.done = True
        observation = self.build_observation(state, robot_pose, object_pose)
        inputs = {
            self.obs_input_name: observation[None, :],
            self.time_input_name: np.array([[0.0]], dtype=np.float32),
        }
        action = self.session.run(None, inputs)[0].squeeze().astype(np.float32)
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"invalid OmniContact action shape/values: {action.shape}")
        self.action_lab = action
        target_lab = action * self.action_scale_lab + self.default_lab
        visualization = self.reference_visualization()
        return PolicyStep(
            command=PDCommand(target_lab, self.kp_lab, self.kd_lab),
            observation=observation,
            task_state="trajectory_complete" if self.done else "executing",
            visualization=visualization,
        )

    def reference_visualization(self) -> ReferenceVisualization:
        if self.reference is None:
            raise RuntimeError("OmniContact reference is not initialized")
        last = len(self.reference["ref_contact"]) - 1
        index = min(self.frame, last)

        def pose(position_key: str, quaternion_key: str) -> np.ndarray:
            return np.concatenate(
                (
                    self.reference[position_key][index],
                    self.reference[quaternion_key][index],
                )
            ).astype(np.float32)

        ghost_base = None
        ghost_dof = None
        if "ref_base_pos" in self.reference and "ref_base_quat" in self.reference:
            ghost_base = pose("ref_base_pos", "ref_base_quat")
        if "dof_pos" in self.reference:
            ghost_dof = np.asarray(self.reference["dof_pos"][index], dtype=np.float32)

        return ReferenceVisualization(
            left_wrist_wxyz=pose("ref_left_wrist_pos", "ref_left_wrist_quat"),
            right_wrist_wxyz=pose("ref_right_wrist_pos", "ref_right_wrist_quat"),
            torso_wxyz=pose("ref_torso_future_pos", "ref_torso_future_quat"),
            left_ankle_wxyz=pose(
                "ref_left_ankle_future_pos", "ref_left_ankle_future_quat"
            ),
            right_ankle_wxyz=pose(
                "ref_right_ankle_future_pos", "ref_right_ankle_future_quat"
            ),
            object_wxyz=pose("ref_object_pos", "ref_object_quat"),
            contact=self.reference["ref_contact"][index],
            ghost_base_wxyz=ghost_base,
            ghost_dof_pos=ghost_dof,
        )

    def advance(self) -> None:
        if self.reference is not None and not self.done:
            self.frame += 1

    def should_replan(self, object_pose: ObjectPose, goal: TaskGoal) -> bool:
        # carrybox-only deployment performs one reference and then holds its
        # final policy frame.  Retrying the entire carry after completion can
        # unexpectedly start a second motion while the operator expects hold.
        if self.reference is None or self.done:
            return False
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1
            return False
        if (
            object_pose.linear_velocity_w is None
            or object_pose.angular_velocity_w is None
        ):
            return False
        if (
            float(np.linalg.norm(object_pose.linear_velocity_w))
            > self.replan_linear_speed
            or float(np.linalg.norm(object_pose.angular_velocity_w))
            > self.replan_angular_speed
        ):
            return False
        reference_pos = self.reference["ref_object_pos"][
            min(self.frame, len(self.reference["ref_object_pos"]) - 1)
        ]
        if (
            float(np.linalg.norm(reference_pos - object_pose.position_w))
            > self.replan_position_error
        ):
            return True
        return False

    def request_replan(
        self,
        robot_pose: RobotPose,
        object_pose: ObjectPose,
        goal: TaskGoal,
    ) -> None:
        if time.monotonic() >= self._next_replan_attempt_s:
            self._start_replan(robot_pose, object_pose, goal)

    @staticmethod
    def _clip_norm(vector: np.ndarray, maximum: float = 4.0) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector * np.minimum(1.0, maximum / np.maximum(norm, 1e-8))

    @staticmethod
    def _gravity(quaternion_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float32).reshape(4)
        return np.array(
            (
                2.0 * (-z * x + w * y),
                -2.0 * (z * y + w * x),
                1.0 - 2.0 * (w * w + z * z),
            ),
            dtype=np.float32,
        )
