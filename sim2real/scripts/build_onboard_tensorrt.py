#!/usr/bin/python3
"""Build engine files on the destination Jetson with its installed TensorRT."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import tensorrt as trt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--robot", choices=["a", "b"], required=True)
    a = p.parse_args()
    root = Path(a.directory).resolve()
    manifest = json.loads((root / "export.json").read_text())
    names = ["scalebfm", f"residual_{a.robot}"]
    engines = {}
    for name in names:
        onnx = root / f"{name}.onnx"
        engine = root / f"{name}.engine"
        if sha(onnx) != manifest["models"][name]["onnx_sha256"]:
            raise ValueError("ONNX checksum mismatch")
        with (root / f"{name}.build.log").open("w") as log:
            build_only = (
                "--skipInference"
                if int(trt.__version__.split(".")[0]) >= 10
                else "--buildOnly"
            )
            subprocess.run(
                [
                    "/usr/src/tensorrt/bin/trtexec",
                    f"--onnx={onnx}",
                    f"--saveEngine={engine}",
                    "--noTF32",
                    build_only,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        engines[engine.name] = sha(engine)
        print("Built", engine, flush=True)
    manifest.update(
        robot=a.robot,
        tensorrt=trt.__version__,
        architecture=platform.machine(),
        engines=engines,
        precision="fp32_no_tf32",
    )
    (root / "engine_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
