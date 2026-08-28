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
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.directory = Path(output_dir).expanduser().resolve() / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.rows: list[dict[str, Any]] = []

    def record(self, coordinator, result) -> None:
        states = [robot.last_state for robot in coordinator.robots]
        snapshot: DualPoseSnapshot | None = coordinator.pose_provider.get_snapshot()
        if any(state is None for state in states) or snapshot is None:
            return
        step = result.policy_step
        commands = [robot.last_command for robot in coordinator.robots]
        self.rows.append(
            {
                "time_ns": time.monotonic_ns(),
                "deployment_state": result.state.value,
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
                "scalebfm_target": np.full((2, 29), np.nan, dtype=np.float32)
                if step is None
                else step.scalebfm_targets,
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
        if not self.rows:
            return None
        keys = tuple(self.rows[0])
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in keys}
        output = self.directory / "rollout.npz"
        np.savez_compressed(output, **arrays)
        self.metadata["ticks"] = len(self.rows)
        self.metadata["saved_at_unix_s"] = time.time()
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )
        return output
