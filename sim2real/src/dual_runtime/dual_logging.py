"""Small logging helper so every dual event carries an explicit robot id."""

from __future__ import annotations

import logging


def robot_logger(robot_id: str) -> logging.LoggerAdapter:
    if robot_id not in {"a", "b"}:
        raise ValueError("robot_id must be 'a' or 'b'")
    return logging.LoggerAdapter(logging.getLogger("dual_g1"), {"robot_id": robot_id})

