#!/usr/bin/env python3
"""Check existing topology and supervise two namespace relays plus team hub."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dual_runtime.onboard_config import load_onboard_config, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/g1/onboard_scalebfm_wired.yaml"))
    args = parser.parse_args()
    path = Path(args.config).expanduser().resolve()
    net = load_onboard_config(path)["network"]
    if net["transport"] != "wired_namespace":
        parser.error("requires wired_namespace transport")
    subprocess.run(["sudo", "-v"], check=True)
    topology = resolve(path.parent, net["wired_topology"])
    subprocess.run(["sudo", "env", f"DUAL_NETWORK_CONFIG={topology}", "bash",
                    str(ROOT / "scripts/setup_dual_network.sh"), "--check"], check=True)
    for side in ("a", "b"):
        namespace = net[side]["namespace"]
        pids = subprocess.check_output(["sudo", "-n", "ip", "netns", "pids", namespace], text=True).split()
        for pid in pids:
            try:
                name = Path(f"/proc/{int(pid)}/comm").read_text().strip()
            except FileNotFoundError:
                continue
            if name == "g1_udp_bridge":
                raise RuntimeError(f"stop the old host g1_udp_bridge in {namespace} before onboard deployment")
    stop = False
    def stop_handler(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    children = []
    try:
        for side in ("a", "b", "hub"):
            prefix = [] if side == "hub" else ["sudo", "-n", "ip", "netns", "exec", net[side]["namespace"]]
            children.append(subprocess.Popen(prefix + [sys.executable,
                str(ROOT / "src/relay_onboard_network.py"), "--config", str(path), "--side", side],
                start_new_session=True))
        print("Wired onboard relays running. Keep this terminal open; Ctrl-C stops all relays.", flush=True)
        while not stop:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("one onboard relay exited; stopping the group")
            time.sleep(.1)
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    subprocess.run(["sudo", "-n", "kill", "-TERM", "--", f"-{child.pid}"], check=False)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "-n", "kill", "-KILL", "--", f"-{child.pid}"], check=False)
                child.wait(timeout=5)


if __name__ == "__main__":
    main()
