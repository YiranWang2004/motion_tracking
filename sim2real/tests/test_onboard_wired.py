import json
from pathlib import Path
import socket
import time

import pytest

from dual_runtime.onboard_config import load_onboard_config, channel_endpoints
from dual_runtime.onboard_network import LatestChannel, PROTOCOL
from dual_runtime.onboard_relay import allowed_packet, namespace_relays, team_hub
from dual_runtime.onboard_team import TeamState


def test_wired_configuration_uses_existing_namespace_topology():
    root = Path(__file__).resolve().parents[1]
    net = load_onboard_config(root / "config/g1/onboard_scalebfm_wired.yaml")["network"]
    assert net["a"]["namespace"] == "g1a"
    assert net["b"]["namespace"] == "g1b"
    assert net["a"]["host"] == net["b"]["host"] == "192.168.123.164"
    assert channel_endpoints(net, "a", "publisher") == (("10.201.1.1", 55300), ("10.201.1.2", 55310))
    assert channel_endpoints(net, "b", "pose") == (("192.168.123.164", 55310), ("192.168.123.201", 55301))
    assert channel_endpoints(net, "a", "team") == (("192.168.123.164", 55320), ("192.168.123.201", 55320))
    wlan = load_onboard_config(root / "config/g1/onboard_scalebfm.yaml")["network"]
    assert channel_endpoints(wlan, "a", "team") == (("192.168.50.11", 55320), ("192.168.50.12", 55320))


def test_relays_reject_joint_states_and_policy_commands():
    team = dict(protocol=PROTOCOL, kind="team", stream="s", seq=1, sent=100.,
                payload=TeamState("a", "f").status(ready=True, frame=1))
    assert allowed_packet(json.dumps(team).encode(), "team")
    assert not allowed_packet(json.dumps(team).encode(), "pose")
    for field in ("q", "dq", "q_des", "kp", "residual", "target"):
        contaminated = {**team, "payload": {**team["payload"], field: [0]*29}}
        assert not allowed_packet(json.dumps(contaminated).encode(), "team")
    assert not allowed_packet(b"\x80\x04binary motor command", "pose")
    assert not allowed_packet(json.dumps({"q_des": [0]*29, "enable": 1}).encode(), "pose")


def wait_read(channel):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        try:
            return channel.read(1.)
        except RuntimeError:
            time.sleep(.01)
    return channel.read(1.)


def loopback_wired_net(new_port):
    # Separate loopback addresses emulate separate interfaces without root/netns.
    net = dict(transport="wired_namespace", max_clock_rtt_s=.02)
    for i, side in enumerate(("a", "b")):
        net[side] = dict(host=f"127.0.0.{10+i}", relay_host=f"127.0.0.{20+i}",
            relay_veth_host=f"127.0.0.{30+i}", publisher_host=f"127.0.0.{40+i}",
            pose_port=new_port(), host_pose_port=new_port(), team_port=new_port())
    return net


def test_wired_clock_and_team_datagrams_survive_all_relay_hops():
    def port():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    net = loopback_wired_net(port)
    relays, channels = [], []
    try:
        for side in ("a", "b"):
            relays.extend(namespace_relays(net, side))
        relays.append(team_hub(net))
        for side in ("a", "b"):
            channels.append(LatestChannel(*channel_endpoints(net, side, "team"), "team"))
        a, b = channels
        sent = TeamState("a", "fingerprint").status(ready=True, frame=7)
        a.publish(sent)
        received, offset, _ = wait_read(b)
        assert received == sent
        assert abs(offset) < .01  # actual A clock, not the host hub clock
        b.publish(TeamState("b", "fingerprint").status(ready=True, frame=7))
        assert wait_read(a)[0]["robot_id"] == "b"
        publisher = LatestChannel(*channel_endpoints(net, "a", "publisher"), "pose")
        receiver = LatestChannel(*channel_endpoints(net, "a", "pose"), "pose")
        channels.extend((publisher, receiver))
        payload = dict(calibration_id="c", snapshot_seq=3, captured=time.time(),
            poses=[[0, 0, 0, 0, 0, 0, 1]]*3, half_extents=[.5, .15, .15])
        publisher.publish(payload)
        assert wait_read(receiver)[0] == payload  # acquisition time unchanged
        assert all(r.error is None for r in relays)
        # Relay failure must not manufacture heartbeats or acknowledge clocks.
        relays[-1].close()
        time.sleep(.14)
        with pytest.raises(RuntimeError):
            a.read(.12)
    finally:
        for c in channels:
            c.close()
        for r in relays:
            r.close()
