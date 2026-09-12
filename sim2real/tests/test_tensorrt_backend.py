"""Check pipe deadlines, artifact contracts and cleanup without a GPU."""

import json
import subprocess
import sys
import numpy as np
import pytest
from dual_runtime import tensorrt_backend as module
from dual_runtime.onboard_config import sha256

HEADER = {
    "tensorrt": "test",
    "models": [
        {"inputs": [[1, 3, 64], [1, 3, 29], [1, 6, 267]], "output": [1, 29]},
        {"inputs": [[1, 201]], "output": [1, 29]},
    ],
}


@pytest.fixture
def directory(tmp_path):
    for name in ["scalebfm.engine", "residual_a.engine"]:
        (tmp_path / name).write_bytes(b"engine")
    (tmp_path / "engine_manifest.json").write_text(
        json.dumps(
            dict(
                sources={},
                robot="a",
                tensorrt="test",
                engines={p.name: sha256(p) for p in tmp_path.glob("*.engine")},
            )
        )
    )
    return tmp_path


def fake_worker(monkeypatch, body):
    original = subprocess.Popen

    def spawn(command, **kwargs):
        return original(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys,time,struct\nprint("
                + repr(json.dumps(HEADER))
                + ",flush=True)\n"
                + body,
            ],
            **kwargs,
        )

    monkeypatch.setattr(module.subprocess, "Popen", spawn)


def test_pipe_roundtrip_and_close(directory, monkeypatch):
    fake_worker(
        monkeypatch,
        """while True:
 index=sys.stdin.buffer.read(1)
 if not index:break
 n=201*4 if index==b'\\x01' else (3*64+3*29+6*267)*4
 data=sys.stdin.buffer.read(n)
 sys.stdout.buffer.write(struct.pack('29f',*([0.25]*29)));sys.stdout.buffer.flush()
""",
    )
    runner = module.TensorRTBackend(directory, {}, "a", timeout_s=0.2)
    process = runner.process
    try:
        np.testing.assert_array_equal(runner.residual(np.zeros(201)), np.full(29, 0.25))
        with pytest.raises(ValueError, match="invalid TensorRT inputs"):
            runner.residual(np.full(201, np.nan))
    finally:
        runner.close()
    assert process.poll() is not None
    runner.close()


def test_worker_deadline_is_bounded(directory, monkeypatch):
    fake_worker(monkeypatch, "time.sleep(30)\n")
    runner = module.TensorRTBackend(directory, {}, "a", timeout_s=0.01)
    try:
        with pytest.raises(RuntimeError, match="timeout"):
            runner.residual(np.zeros(201))
    finally:
        runner.close()


def test_reject_wrong_actor_or_engine_before_spawn(directory, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before spawning")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="another robot"):
        module.TensorRTBackend(directory, {}, "b")
    (directory / "scalebfm.engine").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.TensorRTBackend(directory, {}, "a")
