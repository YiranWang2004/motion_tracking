"""Read-only UDP transport for the sim2real MuJoCo twin."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.udp_latest import UDPLatestReceiver, UDPLatestSender


PROTOCOL = "robojudo.omnicontact.visualization"
VERSION = 1


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(shape)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result.copy()


def encode_visualization(
    scene: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a validated, numpy-backed visualization packet."""
    packet: dict[str, Any] = {"protocol": PROTOCOL, "version": VERSION}
    if scene is not None:
        packet["scene"] = {
            name: _array(scene[name], (7,), name)
            for name in ("start_plane_wxyz", "goal_plane_wxyz")
        }
    if reference is not None:
        result = {
            name: _array(reference[name], (7,), name)
            for name in (
                "left_wrist_wxyz",
                "right_wrist_wxyz",
                "torso_wxyz",
                "left_ankle_wxyz",
                "right_ankle_wxyz",
                "object_wxyz",
            )
        }
        result["contact"] = _array(reference["contact"], (4,), "contact")
        if reference.get("ghost_base_wxyz") is not None:
            result["ghost_base_wxyz"] = _array(
                reference["ghost_base_wxyz"], (7,), "ghost_base_wxyz"
            )
        if reference.get("ghost_dof_pos") is not None:
            result["ghost_dof_pos"] = _array(
                reference["ghost_dof_pos"], (-1,), "ghost_dof_pos"
            )
        packet["reference"] = result
    if len(packet) == 2:
        raise ValueError("visualization packet must contain scene or reference")
    return packet


def decode_visualization(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    try:
        if data.get("protocol") != PROTOCOL or int(data.get("version", -1)) != VERSION:
            return None
        return encode_visualization(data.get("scene"), data.get("reference"))
    except (KeyError, TypeError, ValueError):
        return None


class VisualizationSender:
    """Best-effort latest-packet sender; it never raises into motor control."""

    def __init__(self, host: str, port: int) -> None:
        self._sender = UDPLatestSender(host, int(port))

    def send(self, scene: dict[str, Any] | None, reference: dict[str, Any] | None) -> None:
        try:
            self._sender.send(encode_visualization(scene, reference))
        except (OSError, KeyError, TypeError, ValueError):
            # Visualization is intentionally non-critical to deployment.
            return

    def close(self) -> None:
        self._sender.close()


class VisualizationReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.receiver = UDPLatestReceiver(host, int(port))

    def start(self) -> None:
        self.receiver.start()

    def read_latest(self, *, with_meta: bool = False):
        packet = self.receiver.read_latest_data(with_meta=with_meta)
        if packet is None:
            return None
        if with_meta:
            decoded = decode_visualization(packet.data)
            if decoded is None:
                return None
            return packet.__class__(
                seq=packet.seq,
                send_time_ns=packet.send_time_ns,
                recv_time_ns=packet.recv_time_ns,
                addr=packet.addr,
                data=decoded,
            )
        return decode_visualization(packet)

    def close(self) -> None:
        self.receiver.close()
