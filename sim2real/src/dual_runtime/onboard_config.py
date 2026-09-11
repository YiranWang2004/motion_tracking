"""Portable, role-independent configuration and checked onboard artifacts."""
import hashlib
import json
import ipaddress
from pathlib import Path

import yaml


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def load_onboard_config(path):
    """Expand wired endpoints from the same topology used by setup_dual_network.

    Read configuration only: neither the host nor robot changes interfaces here.
    Robots need a copy of the topology YAML, not network namespaces themselves.
    """
    path = Path(path).expanduser().resolve()
    config = load_yaml(path)
    net = config["network"]
    mode = net.setdefault("transport", "wireless")
    if mode not in {"wireless", "wired_namespace"}:
        raise ValueError("network.transport must be wireless or wired_namespace")
    if mode == "wired_namespace":
        topology = load_yaml(resolve(path.parent, net["wired_topology"]))
        for side in ("a", "b"):
            source = topology[f"robot_{side}"]
            local = net[side]
            local.setdefault("host", topology["robot_ip"])
            local["namespace"] = source["namespace"]
            local["relay_host"] = str(ipaddress.ip_interface(source["robot_address"]).ip)
            local["publisher_host"] = str(ipaddress.ip_interface(source["veth"]["host_address"]).ip)
            local["relay_veth_host"] = str(ipaddress.ip_interface(source["veth"]["namespace_address"]).ip)
    return config


def channel_endpoints(net, side, role):
    """Return (bind, remote) for an end-to-end LatestChannel."""
    local = net[side]
    wired = net.get("transport", "wireless") == "wired_namespace"
    if role == "publisher":
        return ((local["publisher_host"] if wired else net["host"], local["host_pose_port"]),
                (local["relay_veth_host"] if wired else local["host"], local["pose_port"]))
    if role == "pose":
        return ((local["host"], local["pose_port"]),
                (local["relay_host"] if wired else net["host"], local["host_pose_port"]))
    if role == "team":
        other = net["b" if side == "a" else "a"]
        return ((local["host"], local["team_port"]),
                (local["relay_host"], local["team_port"]) if wired
                else (other["host"], other["team_port"]))
    raise ValueError(f"unknown channel role: {role}")


def resolve(base, value):
    return (Path(base) / Path(value).expanduser()).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("task") != "dual_g1_scalebfm_residual_object":
        raise ValueError("onboard supports the offline residual object task only")
    contract = manifest.get("checkpoint_contract", {})
    required = dict(interaction_frame="pelvis", anchor_angular_velocity_frame="reference-anchor",
                    reference_anchor_body="torso_link", actor_observation="current",
                    box_observation="actual", residual_joints="whole-body")
    if any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("onboard needs pelvis/reference-anchor 201-D artifacts; re-export the matching trained checkpoint")
    files = dict(scalebfm_checkpoint="scalebfm_model.pt", scalebfm_metadata="scalebfm_metadata.json",
                 scalebfm_mode_table="scalebfm_mode_table.pt", residual_checkpoint="residual_actor.pt",
                 reference_bundle="reference_bundle.npz", kinematics_xml="g1_29dof_scalebfm.xml")
    result = {}
    for key, name in files.items():
        path = directory / name
        if sha256(path) != manifest.get("files", {}).get(name, {}).get("sha256"):
            raise ValueError(f"artifact checksum mismatch: {path}")
        result[key] = path
    return result, contract


def fingerprint(files, configuration):
    return hashlib.sha256(json.dumps(
        dict(files={k: sha256(v) for k, v in files.items()}, configuration=configuration),
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()
