#!/usr/bin/env python3
"""Host-only OpenVR service for a single pelvis; no robot command connection."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scalebfm_tracking.pose import serve_vive

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--robot", choices=("a", "b"), default="a")
    p.add_argument("--bind", default="tcp://127.0.0.1:28710")
    args = p.parse_args()
    try:
        serve_vive(args.config, args.robot, args.bind)
    except KeyboardInterrupt:
        pass
