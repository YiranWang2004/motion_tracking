#!/usr/bin/env python3
"""Run a namespace pose/team relay or the default-namespace team hub."""
import argparse
import signal
import time

from dual_runtime.onboard_config import load_onboard_config
from dual_runtime.onboard_relay import namespace_relays, team_hub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/g1/onboard_scalebfm_wired.yaml")
    parser.add_argument("--side", choices=("a", "b", "hub"), required=True)
    args = parser.parse_args()
    net = load_onboard_config(args.config)["network"]
    if net["transport"] != "wired_namespace":
        parser.error("relays require wired_namespace transport")
    stop = False
    def stop_handler(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    relays = [team_hub(net)] if args.side == "hub" else namespace_relays(net, args.side)
    print(f"onboard {args.side} relay ready: pose/team/clock only; no DDS or joint commands", flush=True)
    try:
        while not stop:
            for relay in relays:
                if relay.error:
                    raise RuntimeError(relay.error)
            time.sleep(.1)
    finally:
        for relay in relays:
            relay.close()


if __name__ == "__main__":
    main()
