#!/usr/bin/python3
"""Open A/B bridge and policy terminals using namespace SSH connections."""
import argparse
import getpass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from configobj import ConfigObj
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = {name: f"config/g1/onboard_scalebfm_wired_{name}.yaml"
         for name in ("lateral", "lift")}


def robot_interface(args, side):
    return getattr(args, f"robot_interface_{side}", None) or args.robot_interface or "eth0"


def remote_command(args, side, role):
    root = shlex.quote(args.remote_root)
    if role == "bridge":
        return (f"cd {root}/g1_sim2real && "
                "if pgrep -x g1_udp_bridge >/dev/null; then "
                "echo 'A bridge is already running on this robot; stop it before restarting.'; exit 1; fi; "
                f"cd {root}/g1_sim2real && "
                f"G1_NET={shlex.quote(robot_interface(args, side))} bash scripts/run_onboard_bridge.sh")
    policy = ["bash", "scripts/run_onboard_scalebfm.sh", "--config", TASKS[args.task], "--robot", side]
    if args.actuate:
        policy += ["--actuate", "--confirm-actuation", "ENABLE_MOTORS"]
    return (f"cd {root}/sim2real && "
            'export PATH="$HOME/.local/bin:$PATH"; '
            "echo 'Waiting up to 45 seconds for the local bridge command port 55002...'; "
            "ready=0; for attempt in $(seq 1 45); do "
            "if ss -H -lun 'sport = :55002' | grep -q .; then ready=1; break; fi; sleep 1; done; "
            "if [ \"$ready\" != 1 ]; then echo 'Local bridge did not become ready.'; exit 1; fi; "
            f"cd {root}/sim2real && exec {shlex.join(policy)}")


def ssh_command(socket, host, command):
    # Fail if the master has disappeared; never fall back to the default
    # namespace, where the two identical robot IPs are ambiguous.
    return ["ssh", "-S", str(socket), "-o", "BatchMode=yes", "-o", "ProxyCommand=false",
            "-tt", f"unitree@{host}", "bash -lc " + shlex.quote(command)]


def ensure_connection(side, host, namespace, socket):
    target = f"unitree@{host}"
    check = subprocess.run(["ssh", "-S", str(socket), "-O", "check", target],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        subprocess.run(["sudo", "ip", "netns", "exec", namespace,
                        "sudo", "-u", getpass.getuser(), "ssh", "-M", "-S", str(socket),
                        "-o", "ControlPersist=30m", "-o", "ConnectTimeout=10",
                        "-o", f"HostKeyAlias=unitree-g1-{side}", "-fN", target], check=True)
    subprocess.run(["ssh", "-S", str(socket), "-o", "BatchMode=yes", "-o", "ProxyCommand=false",
                    target, "hostname"], check=True)


def write_layout(path, args, endpoints):
    layout = {
        "window": dict(type="Window", parent="", size=[1400, 900]),
        "columns": dict(type="HPaned", parent="window", position="700"),
        "a_rows": dict(type="VPaned", parent="columns", position="450"),
        "b_rows": dict(type="VPaned", parent="columns", position="450"),
    }
    for side in ("a", "b"):
        for role in ("bridge", "policy"):
            host, _, socket = endpoints[side]
            command = [sys.executable, str(Path(__file__).resolve()), "--pane", side, role,
                       "--task", args.task, "--remote-root", args.remote_root,
                       "--robot-interface", robot_interface(args, side), "--host", host, "--socket", str(socket)]
            if args.preview:
                command += ["--preview"]
            if args.actuate:
                command += ["--actuate", "--confirm-actuation", "ENABLE_MOTORS"]
            layout[f"{side}_{role}"] = dict(type="Terminal", parent=f"{side}_rows",
                                           profile="onboard", command=shlex.join(command))
    config = ConfigObj()
    config.filename = str(path)
    config["global_config"] = dict(broadcast_default="off", title_hide_sizetext=True)
    config["keybindings"] = {}
    config["profiles"] = {"onboard": dict(use_system_font=False, font="Monospace 11",
                                          scrollback_lines=10000, exit_action="hold")}
    config["layouts"] = {"onboard": layout}
    config["plugins"] = {}
    config.write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="lateral")
    parser.add_argument("--remote-root", default="/home/unitree/wyr/motion_tracking")
    parser.add_argument("--robot-interface", help="override both robots' onboard DDS interfaces")
    parser.add_argument("--robot-interface-a", help="override A's onboard DDS interface")
    parser.add_argument("--robot-interface-b", help="override B's onboard DDS interface")
    parser.add_argument("--preview", action="store_true", help="open four panes without SSH or robot processes")
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--confirm-actuation", default="")
    parser.add_argument("--pane", nargs=2, metavar=("ROBOT", "ROLE"), help=argparse.SUPPRESS)
    parser.add_argument("--host", help=argparse.SUPPRESS)
    parser.add_argument("--socket", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.actuate and args.confirm_actuation != "ENABLE_MOTORS":
        parser.error("--actuate requires --confirm-actuation ENABLE_MOTORS")
    if args.pane:
        side, role = args.pane
        if side not in ("a", "b") or role not in ("bridge", "policy"):
            parser.error("invalid pane")
        title = f"G1 {side.upper()} | {role} | {args.task}" + (" | PREVIEW" if args.preview else "")
        print(f"\033]0;{title}\007{title}", flush=True)
        command = remote_command(args, side, role)
        if args.preview:
            print("Preview only. No SSH connection or robot process started.\n" + command, flush=True)
            input("Press Enter to finish this preview pane. ")
            return
        try:
            result = subprocess.run(ssh_command(args.socket, args.host, command))
            print(f"\n{title}: SSH process exited with status {result.returncode}.", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.", flush=True)
        input("Press Enter to finish this pane; output remains visible. ")
        return
    if os.geteuid() == 0:
        parser.error("run this launcher as the desktop user, not with sudo")
    terminator = shutil.which("terminator") or str(Path.home()/".local/bin/terminator")
    if not Path(terminator).is_file():
        parser.error("Terminator is not installed")
    if not args.preview and subprocess.run(["pgrep", "-x", "g1_udp_bridge"],
                                           stdout=subprocess.DEVNULL).returncode == 0:
        parser.error("stop the old host g1_udp_bridge processes before onboard deployment")
    config = yaml.safe_load((ROOT/TASKS[args.task]).read_text())
    net = config["network"]
    for side in ("a", "b"):
        field = f"robot_interface_{side}"
        if not getattr(args, field):
            setattr(args, field, args.robot_interface or net[side].get("robot_interface", "eth0"))
    topology = yaml.safe_load((ROOT/TASKS[args.task]).parent.joinpath(net["wired_topology"]).read_text())
    sockets = Path.home()/".ssh/ctl"
    if not args.preview:
        sockets.mkdir(parents=True, exist_ok=True, mode=0o700)
    endpoints = {}
    for side in ("a", "b"):
        host = net[side].get("host", topology["robot_ip"])
        namespace = topology[f"robot_{side}"]["namespace"]
        socket = sockets/f"unitree-g1-{side}-deploy"
        endpoints[side] = (host, namespace, socket)
        if not args.preview:
            print(f"Connecting G1 {side.upper()} via {namespace}...", flush=True)
            ensure_connection(side, host, namespace, socket)
    directory = Path(tempfile.mkdtemp(prefix="g1-terminator-"))
    try:
        write_layout(directory/"config", args, endpoints)
        print(f"Task: {args.task}; actuation: {args.actuate}; preview: {args.preview}", flush=True)
        print("Left: A bridge / A policy. Right: B bridge / B policy.", flush=True)
        print("Host wired relays and Vive publisher use separate terminals.", flush=True)
        subprocess.run([terminator, "--no-dbus", "--config", str(directory/"config"),
                        "--layout", "onboard", "--title", f"Dual G1 — {args.task}"], check=True)
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
