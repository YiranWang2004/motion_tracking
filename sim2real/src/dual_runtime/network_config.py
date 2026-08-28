"""Validated persistent network topology for two same-address G1 robots."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_INTERFACE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,15}$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,63}$")
_MAC_RE = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _name(value: Any, name: str, pattern: re.Pattern[str]) -> str:
    result = str(value).strip()
    if not pattern.fullmatch(result):
        raise ValueError(f"invalid {name}: {result!r}")
    return result


def _interface(value: Any, name: str) -> ipaddress.IPv4Interface:
    try:
        result = ipaddress.ip_interface(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not isinstance(result, ipaddress.IPv4Interface):
        raise ValueError(f"{name} must be IPv4")
    return result


@dataclass(frozen=True)
class DualNetworkSide:
    robot_id: str
    interface: str
    expected_mac: str
    namespace: str
    robot_address: ipaddress.IPv4Interface
    robot_ip: ipaddress.IPv4Address
    veth_host_name: str
    veth_host_address: ipaddress.IPv4Interface
    veth_namespace_name: str
    veth_namespace_address: ipaddress.IPv4Interface
    bridge_config: Path

    def shell_lines(self) -> tuple[str, ...]:
        """Fixed-order, newline-safe values consumed by the shell helper."""

        return (
            self.robot_id,
            self.interface,
            self.namespace,
            str(self.robot_address),
            str(self.robot_ip),
            self.veth_host_name,
            str(self.veth_host_address),
            self.veth_namespace_name,
            str(self.veth_namespace_address),
            str(self.bridge_config),
            self.expected_mac,
        )


@dataclass(frozen=True)
class DualNetworkConfig:
    path: Path
    robot_a: DualNetworkSide
    robot_b: DualNetworkSide

    def side(self, robot_id: str) -> DualNetworkSide:
        if robot_id == "a":
            return self.robot_a
        if robot_id == "b":
            return self.robot_b
        raise ValueError("robot_id must be 'a' or 'b'")


def _load_bridge_udp(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return _mapping(_mapping(raw, "bridge config")["udp"], "bridge udp")
    except (OSError, KeyError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load bridge config {path}: {exc}") from exc


def _build_side(
    robot_id: str,
    raw: dict[str, Any],
    *,
    config_dir: Path,
    robot_ip: ipaddress.IPv4Address,
) -> DualNetworkSide:
    veth = _mapping(raw.get("veth"), f"robot_{robot_id}.veth")
    bridge_value = Path(str(raw.get("bridge_config", ""))).expanduser()
    bridge_config = (
        bridge_value.resolve()
        if bridge_value.is_absolute()
        else (config_dir / bridge_value).resolve()
    )
    if not bridge_config.is_file():
        raise ValueError(f"bridge config does not exist: {bridge_config}")
    side = DualNetworkSide(
        robot_id=robot_id,
        interface=_name(
            raw.get("interface"), f"robot_{robot_id}.interface", _INTERFACE_RE
        ),
        expected_mac=_name(
            str(raw.get("expected_mac", "")).lower(),
            f"robot_{robot_id}.expected_mac",
            _MAC_RE,
        ),
        namespace=_name(
            raw.get("namespace"), f"robot_{robot_id}.namespace", _NAMESPACE_RE
        ),
        robot_address=_interface(
            raw.get("robot_address"), f"robot_{robot_id}.robot_address"
        ),
        robot_ip=robot_ip,
        veth_host_name=_name(
            veth.get("host_name"),
            f"robot_{robot_id}.veth.host_name",
            _INTERFACE_RE,
        ),
        veth_host_address=_interface(
            veth.get("host_address"), f"robot_{robot_id}.veth.host_address"
        ),
        veth_namespace_name=_name(
            veth.get("namespace_name"),
            f"robot_{robot_id}.veth.namespace_name",
            _INTERFACE_RE,
        ),
        veth_namespace_address=_interface(
            veth.get("namespace_address"),
            f"robot_{robot_id}.veth.namespace_address",
        ),
        bridge_config=bridge_config,
    )
    if robot_ip not in side.robot_address.network:
        raise ValueError(
            f"robot_{robot_id}.robot_address and robot_ip are not in one subnet"
        )
    if side.veth_host_address.network != side.veth_namespace_address.network:
        raise ValueError(f"robot_{robot_id} veth addresses are not in one subnet")
    if side.veth_host_address.ip == side.veth_namespace_address.ip:
        raise ValueError(f"robot_{robot_id} veth endpoints use the same IP")

    bridge_udp = _load_bridge_udp(bridge_config)
    expected = {
        "state_host": str(side.veth_host_address.ip),
        "state_mirror_host": str(side.veth_host_address.ip),
        "cmd_bind_host": str(side.veth_namespace_address.ip),
    }
    for field, value in expected.items():
        if str(bridge_udp.get(field, "")) != value:
            raise ValueError(
                f"{bridge_config}:{field} must be {value}, got {bridge_udp.get(field)!r}"
            )
    return side


def load_dual_network_config(path: str | Path) -> DualNetworkConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "network config"
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load network config {config_path}: {exc}") from exc
    if int(raw.get("version", -1)) != 1:
        raise ValueError("dual network config version must be 1")
    try:
        robot_ip = ipaddress.ip_address(str(raw["robot_ip"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("robot_ip must be a valid IPv4 address") from exc
    if not isinstance(robot_ip, ipaddress.IPv4Address):
        raise ValueError("robot_ip must be IPv4")
    robot_a = _build_side(
        "a",
        _mapping(raw.get("robot_a"), "robot_a"),
        config_dir=config_path.parent,
        robot_ip=robot_ip,
    )
    robot_b = _build_side(
        "b",
        _mapping(raw.get("robot_b"), "robot_b"),
        config_dir=config_path.parent,
        robot_ip=robot_ip,
    )
    unique_names = (
        robot_a.interface,
        robot_b.interface,
        robot_a.expected_mac,
        robot_b.expected_mac,
        robot_a.namespace,
        robot_b.namespace,
        robot_a.veth_host_name,
        robot_b.veth_host_name,
        robot_a.veth_namespace_name,
        robot_b.veth_namespace_name,
    )
    if len(set(unique_names)) != len(unique_names):
        raise ValueError(
            "physical interfaces/MACs, namespaces, and veth names must all be unique"
        )
    if robot_a.veth_host_address.network.overlaps(robot_b.veth_host_address.network):
        raise ValueError("robot A/B veth subnets must not overlap")
    return DualNetworkConfig(config_path, robot_a, robot_b)
