"""End-to-end dual ScaleBFM base plus MAPPO residual policy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omnicontact.runtime import BridgeState

from .dual_pose_provider import DualPoseSnapshot
from .kinematics import G1PolicyKinematics, LiveKinematics
from .observation import build_residual_observation
from .reference import DualReferenceBundle, ReferenceFrame
from .residual_policy import ResidualPolicy
from .scalebfm_policy import ScaleBFMHistory, ScaleBFMPolicy


def _quaternion_error_rad(left_xyzw: np.ndarray, right_wxyz: np.ndarray) -> float:
    left = np.asarray(left_xyzw, dtype=np.float64).reshape(4)[[3, 0, 1, 2]]
    right = np.asarray(right_wxyz, dtype=np.float64).reshape(4)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))


@dataclass(frozen=True)
class DualPolicyStep:
    targets: np.ndarray
    scalebfm_targets: np.ndarray
    residuals: np.ndarray
    observations: np.ndarray
    frame: int
    complete: bool
    inference_time_s: float


class DualScaleBFMResidualPolicy:
    """Own all history and reference state for one cooperative rollout."""

    def __init__(
        self,
        *,
        scalebfm_checkpoint: str | Path,
        scalebfm_metadata: str | Path,
        scalebfm_mode_table: str | Path,
        residual_checkpoint: str | Path,
        reference_bundle: str | Path,
        kinematics_xml: str | Path,
        device: str = "cpu",
        inference_precision: str = "fp32",
        control_mode: int = 7,
        future_step: int = 5,
        residual_scale: float = 0.10,
        start_frame: int = 1,
        reference_alignment: str = "xyyaw",
        torch_num_threads: int | None = None,
    ) -> None:
        if not 5 <= int(future_step) <= 33:
            raise ValueError("future_step must be in [5, 33]")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.scalebfm = ScaleBFMPolicy(
            scalebfm_checkpoint,
            scalebfm_metadata,
            scalebfm_mode_table,
            device=device,
            inference_precision=inference_precision,
            torch_num_threads=torch_num_threads,
        )
        self.residual = ResidualPolicy(residual_checkpoint, device=device)
        self.reference = DualReferenceBundle(reference_bundle)
        self.kinematics = (
            G1PolicyKinematics(kinematics_xml),
            G1PolicyKinematics(kinematics_xml),
        )
        self.histories = (ScaleBFMHistory(), ScaleBFMHistory())
        self.previous_residual = np.zeros((2, 29), dtype=np.float32)
        self.previous_executed_action = np.zeros((2, 29), dtype=np.float32)
        self.control_mode = int(control_mode)
        self.future_step = int(future_step)
        self.time_offsets = np.asarray(
            (0, 1, 2, 3, 4, self.future_step), dtype=np.int64
        )
        self.residual_scale = float(residual_scale)
        self.start_frame = int(start_frame)
        if not 0 <= self.start_frame < self.reference.frames:
            raise ValueError("start_frame is outside the reference")
        self.frame = self.start_frame
        self.reference_alignment = reference_alignment
        self.initialized = False

    @property
    def default_q(self) -> np.ndarray:
        return self.scalebfm.default_q

    @property
    def kp(self) -> np.ndarray:
        return self.scalebfm.kp

    @property
    def kd(self) -> np.ndarray:
        return self.scalebfm.kd

    @property
    def torque_limit(self) -> np.ndarray:
        return self.scalebfm.torque_limit

    def initialize(
        self,
        snapshot: DualPoseSnapshot,
        *,
        max_partner_position_error_m: float,
        max_object_position_error_m: float,
        max_box_size_error_m: float,
        max_robot_orientation_error_rad: float,
        max_object_orientation_error_rad: float,
    ) -> dict[str, float]:
        if self.initialized:
            raise RuntimeError("dual policy is already initialized")
        self.reference.align_to_robot_a(snapshot.robot_a, mode=self.reference_alignment)
        reference_a, reference_b = self.reference.frame(self.start_frame)
        partner_error = float(
            np.linalg.norm(snapshot.robot_b.position_w - reference_b.body_pos_w[0])
        )
        object_error = float(
            np.linalg.norm(snapshot.object.position_w - reference_a.object_pos_w)
        )
        box_size_error = float(
            np.max(
                np.abs(snapshot.object.half_extents - self.reference.box_half_extents)
            )
        )
        robot_orientation_error = max(
            _quaternion_error_rad(
                snapshot.robot_a.quaternion_xyzw, reference_a.body_quat_wxyz[0]
            ),
            _quaternion_error_rad(
                snapshot.robot_b.quaternion_xyzw, reference_b.body_quat_wxyz[0]
            ),
        )
        object_orientation_error = _quaternion_error_rad(
            snapshot.object.quaternion_xyzw, reference_a.object_quat_wxyz
        )
        if partner_error > max_partner_position_error_m:
            raise RuntimeError(
                f"robot B initial position differs from reference by {partner_error:.3f} m"
            )
        if object_error > max_object_position_error_m:
            raise RuntimeError(
                f"object initial position differs from reference by {object_error:.3f} m"
            )
        if box_size_error > max_box_size_error_m:
            raise RuntimeError(
                f"box half extents differ from reference by {box_size_error:.3f} m"
            )
        if robot_orientation_error > max_robot_orientation_error_rad:
            raise RuntimeError(
                "robot initial orientation differs from reference by "
                f"{robot_orientation_error:.3f} rad"
            )
        if object_orientation_error > max_object_orientation_error_rad:
            raise RuntimeError(
                "object initial orientation differs from reference by "
                f"{object_orientation_error:.3f} rad"
            )
        self.initialized = True
        return {
            "partner_position_error_m": partner_error,
            "object_position_error_m": object_error,
            "box_size_error_m": box_size_error,
            "robot_orientation_error_rad": robot_orientation_error,
            "object_orientation_error_rad": object_orientation_error,
        }

    def reset_rollout(self) -> None:
        for history in self.histories:
            history.reset()
        self.previous_residual.fill(0.0)
        self.previous_executed_action.fill(0.0)
        self.frame = self.start_frame

    def _live_kinematics(
        self,
        states: tuple[BridgeState, BridgeState],
        snapshot: DualPoseSnapshot,
    ) -> tuple[LiveKinematics, LiveKinematics]:
        return (
            self.kinematics[0].forward(states[0].q_lab, snapshot.robot_a),
            self.kinematics[1].forward(states[1].q_lab, snapshot.robot_b),
        )

    def compute(
        self,
        states: tuple[BridgeState, BridgeState],
        snapshot: DualPoseSnapshot,
    ) -> DualPolicyStep:
        if not self.initialized:
            raise RuntimeError("initialize must be called before policy inference")
        inference_start = time.perf_counter()
        live = self._live_kinematics(states, snapshot)
        for index, state in enumerate(states):
            self.histories[index].update(
                state.quat_wxyz,
                state.gyro,
                state.q_lab,
                state.dq_lab,
                self.previous_executed_action[index],
            )
        target_body_pos, target_body_quat = self.reference.future_key_bodies(
            self.frame, self.time_offsets
        )
        live_body_pos = np.stack([item.body_pos_w for item in live])
        live_body_quat = np.stack([item.body_quat_wxyz for item in live])
        scalebfm_target, raw_action = self.scalebfm.infer_batch(
            self.histories,
            target_body_pos,
            target_body_quat,
            live_body_pos,
            live_body_quat,
            control_mode=self.control_mode,
            time_offsets=self.time_offsets,
        )
        references: tuple[ReferenceFrame, ReferenceFrame] = self.reference.frame(
            self.frame
        )
        observations = np.stack(
            [
                build_residual_observation(
                    state=states[index],
                    reference=references[index],
                    live=live[index],
                    partner_live=live[1 - index],
                    object_pose=snapshot.object,
                    scalebfm_target=scalebfm_target[index],
                    previous_residual=self.previous_residual[index],
                    default_q=self.scalebfm.default_q,
                )
                for index in range(2)
            ]
        )
        residuals = self.residual.infer(observations)
        targets = scalebfm_target + self.residual_scale * residuals
        self.previous_residual = residuals.copy()
        self.previous_executed_action = raw_action + (
            self.residual_scale * residuals / self.scalebfm.action_scale[None]
        )
        if not np.all(np.isfinite(targets)):
            raise RuntimeError("combined dual policy target is non-finite")
        emitted_frame = self.frame
        complete = self.frame >= self.reference.frames - 1
        if not complete:
            self.frame += 1
        return DualPolicyStep(
            targets=targets.astype(np.float32),
            scalebfm_targets=scalebfm_target,
            residuals=residuals,
            observations=observations,
            frame=emitted_frame,
            complete=complete,
            inference_time_s=time.perf_counter() - inference_start,
        )
