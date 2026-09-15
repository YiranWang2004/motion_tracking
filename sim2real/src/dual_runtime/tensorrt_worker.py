"""Compatibility entry point (including system Python on Jetson)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalebfm.tensorrt_worker import main

if __name__ == "__main__":
    main()
