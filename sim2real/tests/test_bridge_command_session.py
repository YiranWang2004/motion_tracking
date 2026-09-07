import time
from types import SimpleNamespace

import numpy as np
import pytest

from common.bridge_session import BridgeCommandSession
from omnicontact.contracts import PDCommand
from omnicontact.runtime import MotionBridgeClient


def status(owner="", epoch=100, latched=False):
    return dict(protocol=1, session_id=owner, epoch=epoch, watchdog_latched=latched)


def test_new_deploy_handshake_and_watchdog_requires_new_instance():
    session = BridgeCommandSession(required=True)
    begin = session.observe(status())
    assert begin == dict(id=session.session_id, epoch=100, begin=True)
    assert session.command_metadata() == {}
    assert session.observe(status(session.session_id, 101)) is None
    assert session.command_metadata()["bridge_session"]["epoch"] == 101
    with pytest.raises(RuntimeError, match="bridge_command_session_fault"):
        session.observe(status(session.session_id, 102, True))
    replacement = BridgeCommandSession(required=True)
    request = replacement.observe(status(session.session_id, 102, True))
    assert request["id"] != session.session_id
    assert request["begin"]


def test_no_silent_fallback_for_hardware_or_loss_of_session():
    with pytest.raises(RuntimeError, match="protocol_missing"):
        BridgeCommandSession(required=True).observe(None)
    assert BridgeCommandSession().observe(None) is None  # existing simulator
    session = BridgeCommandSession()
    session.observe(status())
    session.observe(status(session.session_id, 101))
    with pytest.raises(RuntimeError, match="protocol_missing"):
        session.observe(None)
    with pytest.raises(RuntimeError, match="session_fault"):
        session.observe(status("b" * 32, 102))


class Transport:
    def __init__(self):
        self.packets = []
        self.commands = []

    def read_next_state(self, **kwargs):
        return self.packets.pop(0)

    def send_command(self, **kwargs):
        self.commands.append(kwargs)
        return len(self.commands)


def packet(seq, control):
    return SimpleNamespace(seq=seq, recv_time_ns=time.monotonic_ns(), data={
        "q": np.zeros(29), "dq": np.zeros(29), "quat_wxyz": [1, 0, 0, 0],
        "gyro": [0, 0, 0], "bridge_control": control,
    })


def test_runtime_withholds_ready_until_ack_and_tags_all_commands(monkeypatch):
    transport = Transport()
    monkeypatch.setattr("omnicontact.runtime.UDPRobotHigh", lambda _: transport)
    client = MotionBridgeClient({}, require_bridge_session=True)
    transport.packets.append(packet(1, status()))
    assert client.read_next(0.1) is None
    hello = transport.commands[-1]
    assert hello["enable"] == 0
    for key in ("kp", "kd", "qd_des"):
        assert not np.any(hello[key])
    transport.packets.append(packet(2, status(client.bridge_session.session_id, 101)))
    state = client.read_next(0.1)
    assert state is not None
    client.send(PDCommand(np.ones(29), np.ones(29), np.ones(29)), enable=1, state=state)
    assert transport.commands[-1]["extra_command"]["bridge_session"] == {
        "id": client.bridge_session.session_id, "epoch": 101, "begin": False,
    }
    transport.packets.append(packet(3, status(client.bridge_session.session_id, 102, True)))
    with pytest.raises(RuntimeError, match="bridge_command_session_fault"):
        client.read_next(0.1)
    assert len(transport.commands) == 2  # no automatic re-arm from the running deploy


def test_cpp_session_gate(tmp_path):
    """Exercise the production C++ admission gate, including restarted sequence 0."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "g1_sim2real"
    binary = tmp_path / "command_session_test"
    subprocess.run([
        "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root / "src"),
        str(root / "tests/command_session_test.cpp"), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
