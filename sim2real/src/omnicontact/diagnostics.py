"""Structured, lossless-enough diagnostics for OmniContact deployment runs."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


def provider_update_counts(provider: Any) -> tuple[int, int]:
    """Return monotonically increasing valid/invalid pose-provider counters."""
    for valid_name, invalid_name in (
        ("valid_updates", "invalid_updates"),
        ("valid_packets", "invalid_packets"),
    ):
        if hasattr(provider, valid_name) or hasattr(provider, invalid_name):
            return (
                int(getattr(provider, valid_name, -1)),
                int(getattr(provider, invalid_name, -1)),
            )
    return -1, -1


def _vector(value: Any, size: int) -> np.ndarray:
    if value is None:
        return np.full(size, np.nan, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"diagnostic value has shape {array.shape}, expected {(size,)}")
    return array.copy()


def _command_field(command: Any, field: str) -> np.ndarray:
    if command is None:
        return np.full(29, np.nan, dtype=np.float32)
    return _vector(getattr(command, field, None), 29)


class ObservationHistoryRecorder:
    """Collect policy inputs and command/state context, then atomically save NPZ.

    A normal carry reference is only a few hundred frames, so buffering it in
    memory keeps disk I/O out of the real-time control loop.  ``save`` writes a
    temporary compressed archive and atomically renames it into place.
    """

    SCHEMA_VERSION = 1
    HISTORY_NAMES = (
        "end_effector_positions_torso_frame",
        "base_angular_velocity",
        "projected_gravity",
        "joint_position_error",
        "joint_velocity",
        "previous_policy_action",
        "object_position_heading_frame",
        "object_rotation_6d_heading_frame",
        "object_bbox_heading_frame",
    )
    HISTORY_START = np.asarray((0, 15, 18, 21, 50, 79, 108, 111, 117), dtype=np.int32)
    HISTORY_END = np.asarray((15, 18, 21, 50, 79, 108, 111, 117, 141), dtype=np.int32)

    def __init__(
        self,
        path: str | Path,
        *,
        joint_names: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_names = tuple(str(name) for name in joint_names)
        if len(self.joint_names) != 29:
            raise ValueError("observation recorder requires exactly 29 joint names")
        self.metadata = dict(metadata or {})
        self._rows: dict[str, list[Any]] = {
            name: []
            for name in (
                "event",
                "task_state",
                "wall_time_ns",
                "monotonic_time_ns",
                "policy_frame",
                "pose_pair_valid",
                "state_packet_seq",
                "state_receive_time_ns",
                "state_packet_arrival_ns",
                "q_lab",
                "dq_lab",
                "imu_quaternion_wxyz",
                "imu_gyro",
                "robot_position_w",
                "robot_quaternion_xyzw",
                "robot_pose_age_ms",
                "robot_confidence",
                "object_position_w",
                "object_quaternion_xyzw",
                "object_half_extents",
                "object_linear_velocity_w",
                "object_angular_velocity_w",
                "object_pose_age_ms",
                "object_confidence",
                "provider_valid_count",
                "provider_invalid_count",
                "observation",
                "observation_history",
                "policy_action",
                "raw_target_pos",
                "safe_target_pos",
                "kp",
                "kd",
                "policy_compute_ms",
                "command_gap_ms",
            )
        }
        self._saved = False
        self.disabled = False
        self.recording_error: str | None = None

    @property
    def row_count(self) -> int:
        return min(len(values) for values in self._rows.values())

    @property
    def invalid_pose_rows(self) -> int:
        return sum(
            not bool(value)
            for value in self._rows["pose_pair_valid"][: self.row_count]
        )

    def disable(self, error: BaseException) -> None:
        """Stop collecting rows after an instrumentation-only failure."""
        self.disabled = True
        self.recording_error = repr(error)

    def record(
        self,
        *,
        state: Any,
        provider: Any,
        policy_frame: int,
        event: str,
        task_state: str,
        pose_pair_valid: bool,
        robot_pose: Any = None,
        object_pose: Any = None,
        observation: Any = None,
        observation_history: Any = None,
        policy_action: Any = None,
        raw_command: Any = None,
        safe_command: Any = None,
        policy_compute_ms: float = np.nan,
        command_gap_ms: float = np.nan,
    ) -> None:
        if self.disabled:
            return
        now_s = time.monotonic()
        valid_count, invalid_count = provider_update_counts(provider)

        def append(name: str, value: Any) -> None:
            self._rows[name].append(value)

        append("event", str(event))
        append("task_state", str(task_state))
        append("wall_time_ns", time.time_ns())
        append("monotonic_time_ns", time.monotonic_ns())
        append("policy_frame", int(policy_frame))
        append("pose_pair_valid", bool(pose_pair_valid))
        append("state_packet_seq", int(state.packet_seq))
        append(
            "state_receive_time_ns",
            -1 if state.state_receive_time_ns is None else int(state.state_receive_time_ns),
        )
        append("state_packet_arrival_ns", int(state.packet_arrival_ns))
        append("q_lab", _vector(state.q_lab, 29))
        append("dq_lab", _vector(state.dq_lab, 29))
        append("imu_quaternion_wxyz", _vector(state.quat_wxyz, 4))
        append("imu_gyro", _vector(state.gyro, 3))

        append("robot_position_w", _vector(getattr(robot_pose, "position_w", None), 3))
        append(
            "robot_quaternion_xyzw",
            _vector(getattr(robot_pose, "quaternion_xyzw", None), 4),
        )
        append(
            "robot_pose_age_ms",
            np.nan
            if robot_pose is None
            else float((now_s - robot_pose.stamp_s) * 1000.0),
        )
        append(
            "robot_confidence",
            np.nan if robot_pose is None else float(robot_pose.confidence),
        )
        append("object_position_w", _vector(getattr(object_pose, "position_w", None), 3))
        append(
            "object_quaternion_xyzw",
            _vector(getattr(object_pose, "quaternion_xyzw", None), 4),
        )
        append(
            "object_half_extents",
            _vector(getattr(object_pose, "half_extents", None), 3),
        )
        append(
            "object_linear_velocity_w",
            _vector(getattr(object_pose, "linear_velocity_w", None), 3),
        )
        append(
            "object_angular_velocity_w",
            _vector(getattr(object_pose, "angular_velocity_w", None), 3),
        )
        append(
            "object_pose_age_ms",
            np.nan
            if object_pose is None
            else float((now_s - object_pose.stamp_s) * 1000.0),
        )
        append(
            "object_confidence",
            np.nan if object_pose is None else float(object_pose.confidence),
        )
        append("provider_valid_count", valid_count)
        append("provider_invalid_count", invalid_count)
        append("observation", _vector(observation, 1244))
        if observation_history is None:
            history = np.full((5, 141), np.nan, dtype=np.float32)
        else:
            history = np.asarray(observation_history, dtype=np.float32)
            if history.shape != (5, 141):
                raise ValueError(
                    f"diagnostic history has shape {history.shape}, expected (5, 141)"
                )
            history = history.copy()
        append("observation_history", history)
        append("policy_action", _vector(policy_action, 29))
        append("raw_target_pos", _command_field(raw_command, "target_pos"))
        append("safe_target_pos", _command_field(safe_command, "target_pos"))
        gain_source = safe_command if safe_command is not None else raw_command
        append("kp", _command_field(gain_source, "kp"))
        append("kd", _command_field(gain_source, "kd"))
        append("policy_compute_ms", float(policy_compute_ms))
        append("command_gap_ms", float(command_gap_ms))

    def save(self, *, termination_reason: str) -> Path:
        if self._saved:
            return self.path
        count = self.row_count
        arrays: dict[str, np.ndarray] = {}
        vector_fields = {
            "q_lab",
            "dq_lab",
            "imu_quaternion_wxyz",
            "imu_gyro",
            "robot_position_w",
            "robot_quaternion_xyzw",
            "object_position_w",
            "object_quaternion_xyzw",
            "object_half_extents",
            "object_linear_velocity_w",
            "object_angular_velocity_w",
            "observation",
            "observation_history",
            "policy_action",
            "raw_target_pos",
            "safe_target_pos",
            "kp",
            "kd",
        }
        bool_fields = {"pose_pair_valid"}
        int64_fields = {
            "wall_time_ns",
            "monotonic_time_ns",
            "state_packet_seq",
            "state_receive_time_ns",
            "state_packet_arrival_ns",
            "provider_valid_count",
            "provider_invalid_count",
        }
        int32_fields = {"policy_frame"}
        for name, all_values in self._rows.items():
            values = all_values[:count]
            if name in vector_fields:
                if values:
                    arrays[name] = np.stack(values).astype(np.float32, copy=False)
                else:
                    tail_shape = {
                        "observation": (1244,),
                        "observation_history": (5, 141),
                        "imu_quaternion_wxyz": (4,),
                        "robot_quaternion_xyzw": (4,),
                        "object_quaternion_xyzw": (4,),
                        "imu_gyro": (3,),
                        "robot_position_w": (3,),
                        "object_position_w": (3,),
                        "object_half_extents": (3,),
                        "object_linear_velocity_w": (3,),
                        "object_angular_velocity_w": (3,),
                    }.get(name, (29,))
                    arrays[name] = np.empty((0, *tail_shape), dtype=np.float32)
            elif name in bool_fields:
                arrays[name] = np.asarray(values, dtype=np.bool_)
            elif name in int64_fields:
                arrays[name] = np.asarray(values, dtype=np.int64)
            elif name in int32_fields:
                arrays[name] = np.asarray(values, dtype=np.int32)
            elif name in ("event", "task_state"):
                arrays[name] = np.asarray(values, dtype="U64")
            else:
                arrays[name] = np.asarray(values, dtype=np.float32)

        schema = {
            "version": self.SCHEMA_VERSION,
            "row_axis": "one row per recorded tracking/control event",
            "observation": "exact 1244-D ONNX input; NaN when policy was not evaluated",
            "observation_layout": {
                "tracking_reference": [0, 539],
                "flattened_five_frame_history": [539, 1244],
            },
            "observation_history": "exact policy history after current sample insertion, shape [N,5,141]",
            "quaternion_conventions": {
                "imu_quaternion_wxyz": "wxyz",
                "robot_quaternion_xyzw": "xyzw",
                "object_quaternion_xyzw": "xyzw",
            },
            "nan_semantics": "value was unavailable/not applicable for this event",
        }
        metadata = dict(self.metadata)
        metadata.update(
            {
                "termination_reason": str(termination_reason),
                "row_count": count,
                "invalid_pose_rows": self.invalid_pose_rows,
                "recording_error": self.recording_error,
            }
        )
        arrays.update(
            {
                "schema_version": np.asarray(self.SCHEMA_VERSION, dtype=np.int32),
                "schema_json": np.asarray(json.dumps(schema, ensure_ascii=False)),
                "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)),
                "joint_names": np.asarray(self.joint_names, dtype="U96"),
                "history_feature_names": np.asarray(self.HISTORY_NAMES, dtype="U96"),
                "history_slice_start": self.HISTORY_START,
                "history_slice_end": self.HISTORY_END,
                "future_reference_frames": np.asarray(
                    (0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50), dtype=np.int32
                ),
            }
        )

        temporary = self.path.with_name(self.path.name + ".tmp.npz")
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._saved = True
        LOGGER.info(
            "Observation history saved: %s rows=%d invalid_pose_rows=%d",
            self.path,
            count,
            self.invalid_pose_rows,
        )
        return self.path
