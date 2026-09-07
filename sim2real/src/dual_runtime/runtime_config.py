"""One set of control parameters for both bridge backends."""
from __future__ import annotations

import math


def shared_control_settings(raw: dict) -> dict:
    control = raw.get("control", {})
    settings = {
        "standing_asset_dir": control.get("standing_asset_dir", "omnicontact"),
        "default_pose_duration_s": float(raw.get("default_pose_duration_s", 2.0)),
        "damping_kd": float(control.get("damping_kd", 8.0)),
        "max_tilt_rad": float(control.get("max_tilt_rad", 0.7)),
        "require_button_release": bool(control.get("require_button_release", True)),
        "phase_target_delta": dict(control.get("phase_target_delta", {
            "default_pose": 0.02, "scalebfm_standing": 1.0, "executing": 1.0,
        })),
        "task_safety": dict(raw.get("task_safety", {
            "object_position_z_error_m": 0.10, "object_position_xyz_error_m": 0.30,
        })),
    }
    phase = settings["phase_target_delta"]
    phase.setdefault("scalebfm_standing", phase.get("loco_standing", 1.0))
    phase.setdefault("loco_standing", phase["scalebfm_standing"])  # legacy callers
    for key in ("default_pose_duration_s", "damping_kd", "max_tilt_rad"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for phase in ("default_pose", "loco_standing", "scalebfm_standing", "executing"):
        value = float(settings["phase_target_delta"][phase])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid target delta for {phase}")
        settings["phase_target_delta"][phase] = value
    for key in ("object_position_z_error_m", "object_position_xyz_error_m"):
        value = float(settings["task_safety"][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid task safety threshold {key}")
        settings["task_safety"][key] = value
    return settings
