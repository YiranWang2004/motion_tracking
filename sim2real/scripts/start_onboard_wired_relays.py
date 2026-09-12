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
    # Authenticate before creating detached process groups. sudo's default
    # tty-scoped timestamp cannot be reused by sudo -n after setsid(). Keep
    # the supervisor privileged so shutdown does not depend on a cached sudo
    # timestamp either. Reuse the selected venv interpreter, not root's uv.
    if os.geteuid() != 0:
        os.execvp("sudo", ["sudo", "--", sys.executable, str(Path(__file__).resolve()),
                           "--config", str(path)])
    topology = resolve(path.parent, net["wired_topology"])
    subprocess.run(["bash", str(ROOT / "scripts/setup_dual_network.sh"), "--check"],
                   env={**os.environ, "DUAL_NETWORK_CONFIG": str(topology)}, check=True)
    for side in ("a", "b"):
        namespace = net[side]["namespace"]
        pids = subprocess.check_output(["ip", "netns", "pids", namespace], text=True).split()
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
            prefix = [] if side == "hub" else ["ip", "netns", "exec", net[side]["namespace"]]
            options = {}
            if side == "hub" and "SUDO_UID" in os.environ:
                options = dict(user=int(os.environ["SUDO_UID"]),
                               group=int(os.environ["SUDO_GID"]), extra_groups=[])
            child = subprocess.Popen(prefix + [sys.executable,
                str(ROOT / "src/relay_onboard_network.py"), "--config", str(path), "--side", side],
                start_new_session=True, **options)
            children.append((side, child))
        print("Starting wired onboard relays; wait for a, b and hub to report ready. "
              "Keep this terminal open; Ctrl-C stops all relays.", flush=True)
        while not stop:
            for side, child in children:
                code = child.poll()
                if code is not None:
                    raise RuntimeError(f"onboard {side} relay exited (code {code}); stopping the group")
            time.sleep(.1)
    finally:
        for _, child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)


if __name__ == "__main__":
    main()
