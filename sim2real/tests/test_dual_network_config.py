from pathlib import Path

import pytest
import yaml

from dual_runtime.network_config import load_dual_network_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/g1/dual_network.yaml"


def test_default_dual_network_config_matches_bridge_endpoints():
    config = load_dual_network_config(DEFAULT_CONFIG)
    assert config.robot_a.interface == "enp10s0"
    assert config.robot_b.interface == "enp11s0"
    assert config.robot_a.expected_mac == "bc:fc:e7:b8:96:49"
    assert config.robot_b.expected_mac == "60:cf:84:8a:8f:95"
    assert config.robot_a.namespace == "g1a"
    assert config.robot_b.namespace == "g1b"
    assert str(config.robot_a.robot_ip) == "192.168.123.164"
    assert str(config.robot_b.robot_ip) == "192.168.123.164"
    assert str(config.robot_a.veth_host_address.ip) == "10.201.1.1"
    assert str(config.robot_b.veth_namespace_address.ip) == "10.201.2.2"
    assert len(config.robot_a.shell_lines()) == 11


def _temporary_config(tmp_path: Path) -> tuple[Path, dict]:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for side in ("robot_a", "robot_b"):
        bridge = (DEFAULT_CONFIG.parent / raw[side]["bridge_config"]).resolve()
        raw[side]["bridge_config"] = str(bridge)
    path = tmp_path / "dual_network.yaml"
    return path, raw


def test_network_config_rejects_duplicate_physical_interfaces(tmp_path):
    path, raw = _temporary_config(tmp_path)
    raw["robot_b"]["interface"] = raw["robot_a"]["interface"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must all be unique"):
        load_dual_network_config(path)


def test_network_config_rejects_overlapping_veth_subnets(tmp_path):
    path, raw = _temporary_config(tmp_path)
    raw["robot_b"]["veth"]["host_address"] = "10.201.1.3/24"
    raw["robot_b"]["veth"]["namespace_address"] = "10.201.1.4/24"
    bridge_path = tmp_path / "bridge_b.yaml"
    bridge = yaml.safe_load(
        Path(raw["robot_b"]["bridge_config"]).read_text(encoding="utf-8")
    )
    bridge["udp"]["state_host"] = "10.201.1.3"
    bridge["udp"]["state_mirror_host"] = "10.201.1.3"
    bridge["udp"]["cmd_bind_host"] = "10.201.1.4"
    bridge_path.write_text(yaml.safe_dump(bridge), encoding="utf-8")
    raw["robot_b"]["bridge_config"] = str(bridge_path)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overlap"):
        load_dual_network_config(path)


def test_network_config_rejects_bridge_veth_mismatch(tmp_path):
    path, raw = _temporary_config(tmp_path)
    raw["robot_a"]["veth"]["host_address"] = "10.201.1.9/24"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="state_host must be 10.201.1.9"):
        load_dual_network_config(path)
