"""Version-isolated local TensorRT worker with deadlines and artifact checks."""

import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import time
import numpy as np
import torch
from .onboard_config import sha256


class TensorRTBackend:
    def __init__(
        self, directory, files, side, python="/usr/bin/python3", timeout_s=0.08
    ):
        directory = Path(directory)
        manifest = json.loads((directory / "engine_manifest.json").read_text())
        if manifest["sources"] != {k: sha256(v) for k, v in files.items()}:
            raise ValueError("TensorRT source artifacts changed; re-export and rebuild")
        if manifest["robot"] != side:
            raise ValueError("TensorRT residual engine belongs to another robot")
        paths = [directory / "scalebfm.engine", directory / f"residual_{side}.engine"]
        for path in paths:
            if sha256(path) != manifest["engines"][path.name]:
                raise ValueError("TensorRT engine checksum mismatch")
        self.timeout_s = timeout_s
        self.process = subprocess.Popen(
            [
                python,
                str(Path(__file__).with_name("tensorrt_worker.py")),
                *map(str, paths),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        )
        atexit.register(self.close)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            deadline = time.monotonic() + 30
            header = bytearray()
            while not header.endswith(b"\n"):
                header.extend(self._read(1, deadline))
                if len(header) > 8192:
                    raise RuntimeError("invalid TensorRT worker handshake")
            self.info = json.loads(header)
            if self.info["tensorrt"] != manifest["tensorrt"]:
                raise ValueError("TensorRT version changed; rebuild engines")
            expected = [[[1, 3, 64], [1, 3, 29], [1, 6, 267]], [[1, 201]]]
            for i, model in enumerate(self.info["models"]):
                if model["inputs"] != expected[i] or model["output"] != [1, 29]:
                    raise ValueError("unexpected TensorRT model interface")
            if len(self.info["models"]) != 2:
                raise ValueError("expected two TensorRT models")
        except BaseException:
            self.close()
            raise

    def _read(self, size, deadline):
        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.process.stdout], [], [], remaining)[0]
            ):
                raise RuntimeError("tensorrt_worker_timeout")
            part = os.read(self.process.stdout.fileno(), size - len(data))
            if not part:
                raise RuntimeError("tensorrt_worker_exited")
            data.extend(part)
        return data

    def infer(self, index, *inputs):
        arrays = [np.asarray(x, dtype=np.float32) for x in inputs]
        model = self.info["models"][index]
        if [list(x.shape) for x in arrays] != model["inputs"] or not all(
            np.isfinite(x).all() for x in arrays
        ):
            raise ValueError("invalid TensorRT inputs")
        data = bytes([index]) + b"".join(x.tobytes() for x in arrays)
        deadline = time.monotonic() + self.timeout_s
        offset = 0
        while offset < len(data):
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([], [self.process.stdin], [], remaining)[1]
            ):
                raise RuntimeError("tensorrt_worker_timeout")
            offset += os.write(self.process.stdin.fileno(), data[offset:])
        result = (
            np.frombuffer(self._read(116, deadline), dtype=np.float32)
            .copy()
            .reshape(1, 29)
        )
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite TensorRT output")
        return result

    def scale(self, prop, actions, task):
        return torch.from_numpy(
            self.infer(
                0,
                prop.detach().cpu().numpy(),
                actions.detach().cpu().numpy(),
                task.detach().cpu().numpy(),
            )
        )

    def residual(self, observation):
        return self.infer(1, np.asarray(observation, dtype=np.float32).reshape(1, 201))[
            0
        ]

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        self.process = None
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        if process.stdout:
            process.stdout.close()
        atexit.unregister(self.close)


def attach_tensorrt(policy, directory, files, side, **kwargs):
    if policy.scalebfm.device.type != "cpu" or policy.residual.device.type != "cpu":
        raise ValueError(
            "TensorRT worker handles CUDA; observation preprocessing must stay on CPU"
        )
    backend = TensorRTBackend(directory, files, side, **kwargs)
    policy.scalebfm.accelerated_inference = backend.scale
    policy.residual.accelerated_agent = (policy.index, backend.residual)
    return backend
