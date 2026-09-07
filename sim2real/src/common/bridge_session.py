"""Client side of the persistent G1 bridge command-session handshake."""
from __future__ import annotations

import uuid


class BridgeCommandSession:
    def __init__(self, *, required: bool = False):
        self.required = required
        self.session_id = uuid.uuid4().hex
        self.epoch: int | None = None
        self.pending_epoch: int | None = None

    def observe(self, status: dict | None) -> dict | None:
        """Return a safe begin request, or None when normal commands may flow."""
        if status is None:
            if self.required or self.epoch is not None:
                raise RuntimeError("bridge_session_protocol_missing: rebuild and restart the G1 bridge once")
            return None
        if status.get("protocol") != 1:
            raise RuntimeError("unsupported_bridge_session_protocol")
        epoch = int(status["epoch"])
        owner = status["session_id"]
        latched = bool(status["watchdog_latched"])
        if self.epoch is not None:
            if latched or epoch != self.epoch or owner != self.session_id:
                raise RuntimeError("bridge_command_session_fault: watchdog or session changed; restart deploy to re-arm")
            return None
        if (owner == self.session_id and not latched and
                self.pending_epoch is not None and epoch == self.pending_epoch + 1):
            self.epoch = epoch
            return None
        self.pending_epoch = epoch
        return {"id": self.session_id, "epoch": epoch, "begin": True}

    def command_metadata(self) -> dict:
        if self.epoch is None:
            return {}
        return {"bridge_session": {"id": self.session_id, "epoch": self.epoch, "begin": False}}
