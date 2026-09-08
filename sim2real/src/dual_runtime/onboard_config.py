"""Portable, role-independent configuration and checked onboard artifacts."""
import hashlib
import json
from pathlib import Path

import yaml


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


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
