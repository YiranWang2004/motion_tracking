#!/usr/bin/env python3
"""Create a checksum-locked deployment artifact directory from Dual_G1_MJ."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SIM2REAL_ROOT = HERE.parents[1]
DEFAULT_SOURCE = HERE.parents[3] / "Dual_G1_MJ"
DEFAULT_OUTPUT = SIM2REAL_ROOT / "config/g1/dual_policy_artifacts"
DEFAULT_RUN = "26-08-20_01-42-05-292432_MAPPO"
DEFAULT_RESIDUAL_CHECKPOINT = (
    DEFAULT_SOURCE
    / "results/ablation_collision_omnicontact_hand/omnicontact-hand-1.5kg"
    / "checkpoints/best_agent.pt"
)
DEFAULT_REFERENCE_BUNDLE = DEFAULT_SOURCE / "results/cfgen_batch_128/motion_000010.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_residual(path: Path) -> dict[str, object]:
    modules = torch.load(path, map_location="cpu", weights_only=True)
    details: dict[str, object] = {}
    for agent in ("robot_0", "robot_1"):
        entry = modules.get(agent)
        if not isinstance(entry, dict):
            raise TypeError(f"checkpoint is missing {agent}")
        policy = entry.get("policy", {})
        scaler = entry.get("state_preprocessor", {})
        shapes = {
            "actor_input": tuple(policy["net_container.0.weight"].shape),
            "actor_output": tuple(policy["net_container.6.weight"].shape),
            "running_mean": tuple(scaler["running_mean"].shape),
            "running_variance": tuple(scaler["running_variance"].shape),
        }
        expected = {
            "actor_input": (512, 201),
            "actor_output": (29, 128),
            "running_mean": (201,),
            "running_variance": (201,),
        }
        if shapes != expected:
            raise ValueError(f"unsupported {agent} checkpoint shapes: {shapes}")
        details[agent] = shapes
    for section in ("policy", "state_preprocessor"):
        left = modules["robot_0"][section]
        right = modules["robot_1"][section]
        tensor_keys = {
            key for key, value in left.items() if isinstance(value, torch.Tensor)
        }
        if tensor_keys != {
            key for key, value in right.items() if isinstance(value, torch.Tensor)
        } or any(not torch.equal(left[key], right[key]) for key in tensor_keys):
            raise ValueError(
                f"checkpoint claims shared deployment but {section} differs"
            )
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help="legacy training run directory under <source-root>/dual_g1_scalebfm_residual_object",
    )
    parser.add_argument(
        "--residual-checkpoint",
        default=None,
        help="explicit residual Actor checkpoint; overrides --run",
    )
    parser.add_argument(
        "--reference-bundle",
        default=str(DEFAULT_REFERENCE_BUNDLE),
        help="two-robot CFGen bundle (default: training motion_000000.npz)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    source = Path(args.source_root).expanduser().resolve()
    reference = Path(args.reference_bundle).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    sources = {
        "scalebfm_model.pt": source / "artifacts/scalebfm/model_22200.pt",
        "scalebfm_metadata.json": source
        / "artifacts/scalebfm/model_22200_metadata.json",
        "scalebfm_mode_table.pt": source / "artifacts/scalebfm/mode_table.pt",
        "residual_actor.pt": (
            Path(args.residual_checkpoint).expanduser().resolve()
            if args.residual_checkpoint is not None
            else (
                DEFAULT_RESIDUAL_CHECKPOINT
                if args.run == DEFAULT_RUN and source == DEFAULT_SOURCE
                else source
                / "dual_g1_scalebfm_residual_object"
                / args.run
                / "checkpoints/best_agent.pt"
            )
        ),
        "reference_bundle.npz": reference,
        "g1_29dof_scalebfm.xml": source
        / "src/dual_g1_mj/omnicontact/g1_29dof_kinematics.xml",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing deployment sources:\n" + "\n".join(missing))
    checkpoint_details = validate_residual(sources["residual_actor.pt"])
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source_path in sources.items():
        destination = output / name
        shutil.copy2(source_path, destination)
        files[name] = {
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "source": str(source_path),
        }
    manifest = {
        "schema_version": 1,
        "task": "dual_g1_scalebfm_residual_object",
        "control_frequency_hz": 50,
        "checkpoint_contract": {
            "control_architecture": "decentralized",
            "actor_sharing": "shared",
            "actor_observation": "current",
            "box_observation": "actual",
            "residual_joints": "whole-body",
            "residual_scale": 0.10,
            "agents": checkpoint_details,
        },
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared {len(files)} artifacts in {output}")
    for name, details in files.items():
        print(f"{details['sha256']}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
