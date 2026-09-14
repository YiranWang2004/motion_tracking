"""Single-agent artifacts/configuration, independent of dual reference bundles."""

import hashlib
from pathlib import Path

import yaml


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != 1:
        raise ValueError("unsupported tracking configuration schema")
    names = {
        "checkpoint": "scalebfm_model.pt",
        "metadata": "scalebfm_metadata.json",
        "mode_table": "scalebfm_mode_table.pt",
        "xml": "g1_29dof_scalebfm.xml",
    }
    root = (path.parent / cfg["artifacts"]).resolve()
    files = {k: root / v for k, v in names.items()}
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(file)
    if cfg["pose_mode"] not in ("local", "vive"):
        raise ValueError("pose_mode must be local or vive")
    for key in (
        "state_timeout_s",
        "source_timeout_s",
        "compute_timeout_s",
        "max_target_delta",
        "stand_s",
        "max_tilt_rad",
    ):
        if not 0 < float(cfg[key]) < float("inf"):
            raise ValueError(f"invalid {key}")
    if cfg["compute_timeout_s"] >= 0.2 or cfg["state_timeout_s"] >= 0.2:
        raise ValueError("state/compute deadlines must precede bridge 200ms watchdog")
    if (
        cfg["control_mode"] != int(cfg["control_mode"])
        or not 0 <= int(cfg["control_mode"]) <= 7
    ):
        raise ValueError("invalid control_mode")
    return cfg, files


def load_policy(files, device="cpu"):
    from scalebfm.policy import ScaleBFMPolicy

    return ScaleBFMPolicy(
        files["checkpoint"],
        files["metadata"],
        files["mode_table"],
        device=device,
        torch_num_threads=1,
    )
