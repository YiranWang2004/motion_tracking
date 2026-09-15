#!/usr/bin/env python3
"""Export standalone batch-one ScaleBFM; no reference NPZ or residual artifacts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from scalebfm_tracking.config import load_config, load_policy, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    _, files = load_config(args.config)
    policy = load_policy(files)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scalebfm.onnx"
    inputs = (torch.zeros(1, 3, 64), torch.zeros(1, 3, 29), torch.zeros(1, 6, 267))
    torch.backends.mha.set_fastpath_enabled(False)
    policy.policy.requires_grad_(False).eval()
    with torch.no_grad():
        torch.onnx.export(
            policy.policy,
            inputs,
            str(path),
            input_names=["prop", "actions", "task"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    (out / "export.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "robot": None,
                "sources": {k: sha256(v) for k, v in files.items()},
                "models": {
                    "scalebfm": {
                        "onnx_sha256": sha256(path),
                        "inputs": [list(x.shape) for x in inputs],
                    }
                },
            },
            indent=2,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
