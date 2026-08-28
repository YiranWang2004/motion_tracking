#!/usr/bin/env python3
"""Validate dual_network.yaml and emit one side for shell scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dual_runtime.network_config import load_dual_network_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config/g1/dual_network.yaml")
    )
    parser.add_argument("--side", choices=("a", "b"), default=None)
    args = parser.parse_args()
    config = load_dual_network_config(args.config)
    if args.side is None:
        print(f"OK {config.path}")
        for side in (config.robot_a, config.robot_b):
            print(
                f"robot_{side.robot_id}: interface={side.interface} "
                f"namespace={side.namespace} robot={side.robot_ip} "
                f"bridge={side.bridge_config}"
            )
        return 0
    print("\n".join(config.side(args.side).shell_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
