#!/usr/bin/env python3
"""Export static batch=1 ScaleBFM and frozen residual actors for TensorRT."""

from dual_runtime.residual_policy import ResidualPolicy
from dual_runtime.scalebfm_policy import ScaleBFMPolicy
from dual_runtime.onboard_config import load_artifacts, sha256
import argparse
import json
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FrozenResidual(nn.Module):
    def __init__(self, policy, index):
        super().__init__()
        self.actor = policy.actors[index]
        self.register_buffer("mean", policy.means[index])
        self.register_buffer("scale", torch.sqrt(policy.variances[index]) + 1.0e-8)

    def forward(self, observation):
        return self.actor(
            ((observation.clamp(-100.0, 100.0) - self.mean) / self.scale).clamp(
                -5.0, 5.0
            )
        ).clamp(-1.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    files, contract = load_artifacts(args.artifacts)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scale = ScaleBFMPolicy(
        files["scalebfm_checkpoint"],
        files["scalebfm_metadata"],
        files["scalebfm_mode_table"],
        torch_num_threads=1,
    )
    residual = ResidualPolicy(files["residual_checkpoint"])
    models = [
        (
            "scalebfm",
            scale.policy,
            (torch.zeros(1, 3, 64), torch.zeros(1, 3, 29), torch.zeros(1, 6, 267)),
            ["prop", "actions", "task"],
        )
    ]
    models += [
        (
            f"residual_{s}",
            FrozenResidual(residual, i),
            (torch.zeros(1, 201),),
            ["observation"],
        )
        for i, s in enumerate("ab")
    ]
    torch.backends.mha.set_fastpath_enabled(False)
    manifest = dict(
        schema=1,
        contract=contract,
        sources={k: sha256(v) for k, v in files.items()},
        models={},
    )
    for name, model, inputs, names in models:
        model.requires_grad_(False).eval()
        path = out / f"{name}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                str(path),
                input_names=names,
                output_names=["output"],
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
        manifest["models"][name] = dict(
            onnx_sha256=sha256(path), inputs=[list(x.shape) for x in inputs]
        )
        print("Exported", path, flush=True)
    (out / "export.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
