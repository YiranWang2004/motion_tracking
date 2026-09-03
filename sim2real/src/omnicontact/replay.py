"""Validated loading and time control for OmniContact diagnostic replays."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


class OmniContactReplayLog:
    """The runtime fields needed to reproduce one deploy trajectory visually."""

    REQUIRED_FIELDS = {
        "event": (),
        "task_state": (),
        "monotonic_time_ns": (),
        "wall_time_ns": (),
        "policy_frame": (),
        "pose_pair_valid": (),
        "state_packet_seq": (),
        "q_lab": (29,),
        "dq_lab": (29,),
        "robot_position_w": (3,),
        "robot_quaternion_xyzw": (4,),
        "object_position_w": (3,),
        "object_quaternion_xyzw": (4,),
        "object_linear_velocity_w": (3,),
        "object_angular_velocity_w": (3,),
    }
    TRACKER_FIELDS = (
        "robot_tracker_position_w",
        "robot_tracker_quaternion_xyzw",
        "object_tracker_position_w",
        "object_tracker_quaternion_xyzw",
    )

    def __init__(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        self.path = path
        self.arrays = arrays
        self.schema_version = int(np.asarray(arrays.get("schema_version", 1)).item())
        self.metadata = json.loads(str(np.asarray(arrays.get("metadata_json", "{}"))))
        self.joint_names = tuple(str(value) for value in arrays["joint_names"])
        if len(self.joint_names) != 29 or len(set(self.joint_names)) != 29:
            raise ValueError("replay log must contain 29 unique joint_names")

        self.frame_count = int(arrays["q_lab"].shape[0])
        if self.frame_count <= 0:
            raise ValueError("replay log contains no runtime frames")
        for name, tail_shape in self.REQUIRED_FIELDS.items():
            array = arrays[name]
            expected = (self.frame_count, *tail_shape)
            if array.shape != expected:
                raise ValueError(
                    f"replay field {name!r} has shape {array.shape}, expected {expected}"
                )

        timestamps = np.asarray(arrays["monotonic_time_ns"], dtype=np.int64)
        elapsed = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
        self.elapsed_s = np.maximum.accumulate(elapsed)
        self.duration_s = float(self.elapsed_s[-1])
        tracker_shapes_valid = all(
            name in arrays and arrays[name].shape == (self.frame_count, 3 if "position" in name else 4)
            for name in self.TRACKER_FIELDS
        )
        self.has_recorded_trackers = bool(
            tracker_shapes_valid
            and "tracker_diagnostics_exact" in arrays
            and np.any(np.asarray(arrays["tracker_diagnostics_exact"], dtype=np.bool_))
        )

    @classmethod
    def load(cls, path: str | Path) -> "OmniContactReplayLog":
        requested_path = Path(path).expanduser().resolve()
        replay_path = (
            requested_path.with_suffix(".observations.npz")
            if requested_path.suffix == ".log"
            else requested_path
        )
        if not replay_path.is_file():
            if requested_path.suffix == ".log":
                raise ValueError(
                    "text logs do not contain replay state; missing structured "
                    f"companion: {replay_path}"
                )
            raise ValueError(f"replay log does not exist: {replay_path}")
        wanted = set(cls.REQUIRED_FIELDS) | {
            "schema_version",
            "metadata_json",
            "joint_names",
            "tracker_diagnostics_exact",
            "tracker_sample_wall_time_ns",
            *cls.TRACKER_FIELDS,
        }
        try:
            with np.load(replay_path, allow_pickle=False) as archive:
                missing = set(cls.REQUIRED_FIELDS) - set(archive.files)
                if "joint_names" not in archive.files:
                    missing.add("joint_names")
                if missing:
                    raise ValueError(
                        "replay log is missing required fields: "
                        + ", ".join(sorted(missing))
                    )
                arrays = {
                    name: np.asarray(archive[name]).copy()
                    for name in archive.files
                    if name in wanted
                }
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("replay log"):
                raise
            raise ValueError(f"cannot load replay log {replay_path}: {exc}") from exc
        return cls(replay_path, arrays)

    def joint_columns(self, requested_names: Iterable[str]) -> np.ndarray:
        index = {name: column for column, name in enumerate(self.joint_names)}
        requested = tuple(str(name) for name in requested_names)
        missing = [name for name in requested if name not in index]
        if missing:
            raise ValueError(
                "replay joint_names do not match the viewer model: " + ", ".join(missing)
            )
        return np.asarray([index[name] for name in requested], dtype=np.int32)

    def frame_index_at(self, elapsed_s: float) -> int:
        return int(
            np.clip(
                np.searchsorted(self.elapsed_s, float(elapsed_s), side="right") - 1,
                0,
                self.frame_count - 1,
            )
        )


class ReplayClock:
    """Wall-clock synchronized replay with pause, stepping, restart and looping."""

    SPEED_STEPS = (0.25, 0.5, 1.0, 5.0, 10.0)

    def __init__(
        self,
        replay: OmniContactReplayLog,
        *,
        speed: float = 1.0,
        start_frame: int = 0,
        paused: bool = False,
        loop: bool = False,
    ) -> None:
        if speed <= 0.0:
            raise ValueError("replay speed must be positive")
        if not 0 <= start_frame < replay.frame_count:
            raise ValueError(
                f"start frame must be in [0, {replay.frame_count - 1}]"
            )
        self.replay = replay
        self.speed = float(speed)
        self.loop = bool(loop)
        self.index = int(start_frame)
        self.paused = bool(paused)
        self.finished = False
        self._anchor_wall_s = time.monotonic()
        self._anchor_log_s = float(replay.elapsed_s[self.index])
        self._lock = threading.Lock()

    def update(self, now_s: float | None = None) -> int:
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            if self.paused:
                return self.index
            target = self._anchor_log_s + (now - self._anchor_wall_s) * self.speed
            if target >= self.replay.duration_s:
                if self.loop and self.replay.duration_s > 0.0:
                    target %= self.replay.duration_s
                    self._anchor_wall_s = now
                    self._anchor_log_s = target
                    self.finished = False
                else:
                    self.index = self.replay.frame_count - 1
                    self.paused = True
                    self.finished = True
                    return self.index
            self.index = self.replay.frame_index_at(target)
            return self.index

    def toggle_pause(self, now_s: float | None = None) -> bool:
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            if self.paused:
                self.paused = False
                self.finished = False
                self._anchor_wall_s = now
                self._anchor_log_s = float(self.replay.elapsed_s[self.index])
            else:
                target = self._anchor_log_s + (now - self._anchor_wall_s) * self.speed
                self.index = self.replay.frame_index_at(target)
                self.paused = True
            return self.paused

    def set_speed(self, speed: float, now_s: float | None = None) -> float:
        """Change speed without discontinuously moving the replay position."""
        if speed <= 0.0:
            raise ValueError("replay speed must be positive")
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            if self.paused:
                target = float(self.replay.elapsed_s[self.index])
            else:
                target = self._anchor_log_s + (
                    now - self._anchor_wall_s
                ) * self.speed
                if target >= self.replay.duration_s:
                    if self.loop and self.replay.duration_s > 0.0:
                        target %= self.replay.duration_s
                        self.finished = False
                    else:
                        target = self.replay.duration_s
                        self.paused = True
                        self.finished = True
                self.index = self.replay.frame_index_at(target)
            self.speed = float(speed)
            self._anchor_wall_s = now
            self._anchor_log_s = target
            return self.speed

    def shift_speed(self, direction: int, now_s: float | None = None) -> float:
        """Move to the adjacent fixed speed step and return the selected speed."""
        if direction == 0:
            return self.speed
        if direction < 0:
            candidates = [step for step in self.SPEED_STEPS if step < self.speed]
            speed = candidates[-1] if candidates else self.SPEED_STEPS[0]
        else:
            candidates = [step for step in self.SPEED_STEPS if step > self.speed]
            speed = candidates[0] if candidates else self.SPEED_STEPS[-1]
        return self.set_speed(speed, now_s=now_s)

    def step(self, delta: int, now_s: float | None = None) -> int:
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            self.index = int(np.clip(self.index + int(delta), 0, self.replay.frame_count - 1))
            self.paused = True
            self.finished = self.index == self.replay.frame_count - 1
            self._anchor_wall_s = now
            self._anchor_log_s = float(self.replay.elapsed_s[self.index])
            return self.index

    def restart(self, now_s: float | None = None, *, paused: bool = False) -> int:
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            self.index = 0
            self.paused = bool(paused)
            self.finished = False
            self._anchor_wall_s = now
            self._anchor_log_s = float(self.replay.elapsed_s[0])
            return self.index


def format_replay_progress(
    replay: OmniContactReplayLog,
    index: int,
    *,
    paused: bool,
    speed: float,
    width: int = 28,
) -> str:
    fraction = (index + 1) / replay.frame_count
    filled = min(width, max(0, int(round(width * fraction))))
    bar = "█" * filled + "·" * (width - filled)
    event = str(replay.arrays["event"][index])
    task = str(replay.arrays["task_state"][index])
    pose = "valid" if bool(replay.arrays["pose_pair_valid"][index]) else "invalid"
    state = "PAUSED" if paused else "PLAY"
    wall_time = datetime.fromtimestamp(
        int(replay.arrays["wall_time_ns"][index]) * 1e-9
    ).astimezone().strftime("%H:%M:%S.%f")[:-3]
    return (
        f"[{bar}] {index + 1}/{replay.frame_count} {fraction * 100:6.2f}% "
        f"t={replay.elapsed_s[index]:8.3f}/{replay.duration_s:.3f}s wall={wall_time} "
        f"seq={int(replay.arrays['state_packet_seq'][index])} "
        f"policy={int(replay.arrays['policy_frame'][index])} "
        f"event={event} task={task} pose={pose} {state} {speed:g}x"
    )
