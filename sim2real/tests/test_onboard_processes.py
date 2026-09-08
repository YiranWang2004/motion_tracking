"""Two production deploy processes, actual networks, simulated local bridges.

No external robot addresses or motors are used. Existing weights exercise the
runtime only; the temporary manifest is explicitly a test fixture, not an
assertion that those old weights were trained with the new convention.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

from common.udp_transport import UDPRobotLow
from dual_runtime.onboard_config import sha256
from dual_runtime.onboard_network import LatestChannel


def test_two_onboard_processes_complete_and_stop_on_vive_loss(tmp_path):
    root = Path(__file__).resolve().parents[1]
    old = root / "config/g1/dual_policy_artifacts_contact_v2_8192"
    artifacts = tmp_path / "test_artifacts"
    artifacts.mkdir()
    manifest = json.loads((old / "manifest.json").read_text())
    for name in manifest["files"]:
        (artifacts / name).symlink_to(old / name)
    # A short stationary reference makes this a transport/lifecycle test.
    ref = artifacts / "reference_bundle.npz"
    ref.unlink()
    with np.load(old / "reference_bundle.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    for key in arrays:
        if key.startswith("training_robot_") and any(key.endswith(s) for s in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_ang_vel_w", "body_lin_vel_w")):
            arrays[key] = np.repeat(arrays[key][1:2], 15, axis=0)
        if key in {"training_object_body_pos_w", "training_object_body_quat_w"}:
            arrays[key] = np.repeat(arrays[key][1:2], 15, axis=0)
    np.savez(ref, **arrays)
    manifest["files"][ref.name]["sha256"] = sha256(ref)
    manifest["checkpoint_contract"].update(interaction_frame="pelvis",
        anchor_angular_velocity_frame="reference-anchor", reference_anchor_body="torso_link")
    (artifacts / "manifest.json").write_text(json.dumps(manifest))
    config = yaml.safe_load((root / "config/g1/onboard_scalebfm.yaml").read_text())
    base = yaml.safe_load((root / "config/g1/dual_scalebfm_contact_v2_8192.yaml").read_text())
    base["default_pose_duration_s"] = .2
    base["torch_num_threads"] = 1
    base["joint_limits_source"] = str(root / "config/g1/omnicontact/OmniContact.yaml")
    base["control"]["standing_asset_dir"] = str(root / "config/g1/omnicontact")
    base_file = tmp_path / "policy.yaml"
    base_file.write_text(yaml.safe_dump(base))
    config.update(policy_config=str(base_file), artifact_directory=str(artifacts),
                  vive_config=str(root / "config/g1/omnicontact_vive_dual.json"))
    config["runtime"].update(max_processing_s=.1, max_slow_ticks=10,
        local_state_timeout_s=.1, max_command_gap_s=.15, max_frame_skew=5)
    config["network"]["host"] = "127.0.0.1"
    config["network"]["max_clock_rtt_s"] = .02
    reserved = []
    def new_port():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        reserved.append(sock)
        return sock.getsockname()[1]
    for side in ("a", "b"):
        local = config["network"][side]
        local["host"] = "127.0.0.1"
        for k in ("host_pose_port", "pose_port", "team_port"):
            local[k] = new_port()
        for k in ("state_port", "cmd_port"):
            local["bridge_udp"][k] = new_port()
    for sock in reserved:
        sock.close()
    config_file = tmp_path / "onboard.yaml"
    config_file.write_text(yaml.safe_dump(config))
    calibration_id = sha256(config["vive_config"])
    ports = config["network"]
    channels, bridges, procs, outputs = [], [], [], []
    buttons = [{}, {}]
    quit_event = threading.Event()
    publish_poses = threading.Event()
    publish_poses.set()
    owners, epochs = ["", ""], [100, 100]
    commands = [[], []]
    errors = []
    poses = [np.concatenate((arrays[f"training_robot_{i}_body_pos_w"][0, 0],
                arrays[f"training_robot_{i}_body_quat_w"][0, 0][[1, 2, 3, 0]])).tolist() for i in range(2)]
    poses.append(np.concatenate((arrays["training_object_body_pos_w"][0],
                 arrays["training_object_body_quat_w"][0][[1, 2, 3, 0]])).tolist())
    def server():
        try:
            seq = 0
            while not quit_event.is_set():
                seq += 1
                for i, bridge in enumerate(bridges):
                    command = bridge.read_latest_command()
                    if command:
                        session = command.get("extra_command", {}).get("bridge_session", {})
                        if session.get("begin") and session.get("epoch") == epochs[i]:
                            owners[i] = session["id"]
                            epochs[i] += 1
                        commands[i].append(command)
                    bridge.send_state(
                        q=arrays[f"training_robot_{i}_joint_pos"][0], dq=np.zeros(29),
                        quat_wxyz=arrays[f"training_robot_{i}_body_quat_w"][0, 0],
                        gyro=np.zeros(3), linacc=np.zeros(3), buttons=buttons[i], sticks={},
                        extra_state={"bridge_control": dict(protocol=1, session_id=owners[i],
                            epoch=epochs[i], watchdog_latched=False)})
                    if publish_poses.is_set():
                        channels[i].publish(dict(calibration_id=calibration_id, captured=time.time(),
                            snapshot_seq=seq, poses=poses, half_extents=arrays["training_box_half_extents"].tolist()))
                time.sleep(.01)
        except BaseException as exc:
            errors.append(exc)

    thread = None
    try:
        for side in ("a", "b"):
            local = ports[side]
            channels.append(LatestChannel(("127.0.0.1", local["host_pose_port"]),
                ("127.0.0.1", local["pose_port"]), "pose"))
            udp = local["bridge_udp"]
            bridges.append(UDPRobotLow(dict(state_host="127.0.0.1", state_port=udp["state_port"],
                cmd_bind_host="127.0.0.1", cmd_port=udp["cmd_port"])))
        thread = threading.Thread(target=server)
        thread.start()
        for side in ("a", "b"):
            out = (tmp_path / f"{side}.txt").open("w")
            outputs.append(out)
            procs.append(subprocess.Popen([sys.executable, "-u", "src/deploy_onboard_scalebfm.py",
                "--config", str(config_file), "--robot", side, "--duration", "20",
                "--log-dir", str(tmp_path / "logs")], cwd=root, stdout=out,
                stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(root / "src")}))
        def wait_phase(phase):
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                text = [(tmp_path / f"{s}.txt").read_text() for s in ("a", "b")]
                assert not errors, errors
                assert all(p.poll() is None for p in procs), "\n".join(text)
                if all(f"phase={phase}" in t for t in text):
                    return
                time.sleep(.02)
            raise AssertionError("\n".join(text))
        wait_phase("zero")
        for button, phase in (("start", "default"), ("B", "standing"), ("A", "finished")):
            time.sleep(.3)
            buttons[0] = {button: True}
            time.sleep(.12)
            buttons[0] = {}
            wait_phase(phase)
        publish_poses.clear()
        for proc in procs:
            proc.wait(timeout=5)
        metadata = [json.loads(p.read_text()) for p in (tmp_path / "logs").glob("*/metadata.json")]
        assert len(metadata) == 2
        assert all("pose" in m["exit_reason"] or "peer_fault" in m["exit_reason"] for m in metadata), metadata
        assert metadata[0]["identity"] == metadata[1]["identity"]
        assert metadata[0]["run_id"] == metadata[1]["run_id"]
        for side_commands in commands:
            assert side_commands
            assert all(c["enable"] == 0 for c in side_commands)
            assert all(not np.any(c["kp"]) for c in side_commands)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        quit_event.set()
        if thread:
            thread.join(timeout=2)
        for bridge in bridges:
            bridge.close()
        for channel in channels:
            channel.close()
        for out in outputs:
            out.close()
