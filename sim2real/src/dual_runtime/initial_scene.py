"""One-shot calibrated scene capture; no live hardware dependency after capture."""
from __future__ import annotations

import time
import numpy as np

from .vive_dual_pose import DualViveDeploymentConfig, DualVivePoseProvider


def capture_vive_initial_scene(config_path, *, require_object=True, poll_hz=100.0,
                               wait_timeout_s=10.0, max_age_s=0.1):
    values = np.asarray([poll_hz, wait_timeout_s, max_age_s])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Vive frequency, timeout and maximum pose age must be positive and finite")
    config = DualViveDeploymentConfig.load(config_path, require_object=require_object)
    provider = DualVivePoseProvider(config, poll_hz=poll_hz, require_object=require_object)
    try:
        provider.start()
        if not provider.wait_until_ready(wait_timeout_s):
            raise RuntimeError(f"no complete dual Vive scene within {wait_timeout_s:g}s")
        snapshot = provider.get_snapshot()
        if snapshot is None:
            raise RuntimeError("dual Vive scene disappeared before capture")
        poses = [snapshot.robot_a, snapshot.robot_b]
        if require_object:
            poses.append(snapshot.object)
        now = time.monotonic()
        if any(not 0 <= now - pose.stamp_s <= max_age_s for pose in poses):
            raise RuntimeError(f"dual Vive scene is older than {max_age_s:g}s")
        return snapshot
    finally:
        provider.stop()
