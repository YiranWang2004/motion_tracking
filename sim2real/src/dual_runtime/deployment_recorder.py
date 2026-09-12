"""Crash-tolerant in-memory recorder for one dual policy rollout."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dual_pose_provider import DualPoseSnapshot


class DualDeploymentRecorder:
    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]):
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}"
        self.directory = Path(output_dir).expanduser().resolve() / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.metadata["command_diagnostics_schema_version"] = 1
        self.rows: list[dict[str, Any]] = []

    def record(self, coordinator, result) -> None:
        states = [robot.last_state for robot in coordinator.robots]
        snapshot: DualPoseSnapshot | None = getattr(coordinator, "last_snapshot", None)
        if snapshot is None:
            snapshot = coordinator.pose_provider.get_snapshot()
        self.metadata["last_result"] = {"ok": result.ok, "state": result.state.value, "reason": result.reason}
        alignment = getattr(coordinator, "reference_alignment_pose", None)
        if alignment is not None:
            self.metadata["reference_alignment_pose"] = {"position_w": alignment.position_w.tolist(), "quaternion_xyzw": alignment.quaternion_xyzw.tolist()}
        if any(state is None for state in states) or snapshot is None:
            return
        step = result.policy_step
        commands = [robot.last_command for robot in coordinator.robots]
        diagnostics = [getattr(robot, "last_command_diagnostics", {}) for robot in coordinator.robots]

        def diagnostic_array(key, *, boolean=False):
            return np.stack([
                np.asarray(item.get(key, np.full(29, False if boolean else np.nan)),
                           dtype=bool if boolean else np.float32).copy()
                for item in diagnostics
            ])

        scalebfm_target = np.full((2, 29), np.nan, dtype=np.float32)
        if step is not None:
            scalebfm_target = step.scalebfm_targets
        elif result.ok and result.reason == "scalebfm_default_pose_standing":
            # Standing sends the raw ScaleBFM target directly to RobotSession.
            # Capture it before disabled-robot substitution or either limiter.
            scalebfm_target = diagnostic_array("command_requested_target")
        self.rows.append(
            {
                "time_ns": time.monotonic_ns(),
                "deployment_state": result.state.value,
                "reason": result.reason,
                "ok": result.ok,
                "snapshot_valid": getattr(coordinator, "snapshot_valid", result.ok),
                "processing_time_s": getattr(result, "processing_time_s", 0.0),
                "state_arrival_ns": [state.packet_arrival_ns for state in states],
                "buttons": [[state.buttons.get(key, False) for key in ("start", "B", "A", "stop")] for state in states],
                "command_enable": [getattr(robot, "last_enable", 0) for robot in coordinator.robots],
                "command_kp": np.stack([np.zeros(29) if command is None or command.kp is None else command.kp for command in commands]),
                "command_kd": np.stack([np.zeros(29) if command is None or command.kd is None else command.kd for command in commands]),
                "command_diagnostics_valid": [bool(item) for item in diagnostics],
                "command_requested_target": diagnostic_array("command_requested_target"),
                "command_pre_limit_target": diagnostic_array("command_pre_limit_target"),
                "command_post_target_limit": diagnostic_array("command_post_target_limit"),
                "torque_limit_enabled": [item.get("torque_limit_enabled", False) for item in diagnostics],
                "torque_limit_triggered": diagnostic_array("torque_limit_triggered", boolean=True),
                "estimated_torque_pre_limit": diagnostic_array("estimated_torque_pre_limit"),
                "estimated_torque_clipped": diagnostic_array("estimated_torque_clipped"),
                "q": np.stack([state.q_lab for state in states]),
                "dq": np.stack([state.dq_lab for state in states]),
                "imu_quat_wxyz": np.stack([state.quat_wxyz for state in states]),
                "gyro": np.stack([state.gyro for state in states]),
                "robot_position_w": np.stack(
                    (snapshot.robot_a.position_w, snapshot.robot_b.position_w)
                ),
                "robot_quat_xyzw": np.stack(
                    (snapshot.robot_a.quaternion_xyzw, snapshot.robot_b.quaternion_xyzw)
                ),
                "object_position_w": snapshot.object.position_w,
                "object_quat_xyzw": snapshot.object.quaternion_xyzw,
                "object_half_extents": snapshot.object.half_extents,
                "command_target": np.stack(
                    [
                        np.full(29, np.nan, dtype=np.float32)
                        if command is None
                        else command.target_pos
                        for command in commands
                    ]
                ),
                "frame": -1 if step is None else step.frame,
                "scalebfm_target": scalebfm_target,
                "residual": np.full((2, 29), np.nan, dtype=np.float32)
                if step is None
                else step.residuals,
                "observation": np.full((2, 201), np.nan, dtype=np.float32)
                if step is None
                else step.observations,
                "inference_time_s": np.nan if step is None else step.inference_time_s,
            }
        )

    def save(self) -> Path | None:
        keys = tuple(self.rows[0]) if self.rows else ()
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in keys}
        output = self.directory / "rollout.npz"
        temporary = self.directory / "rollout.partial.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
        self.metadata["ticks"] = len(self.rows)
        self.metadata["saved_at_unix_s"] = time.time()
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )
        return output
