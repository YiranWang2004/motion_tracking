"""Nonblocking two-robot phase agreement; time is in robot A's clock domain."""
from __future__ import annotations

import math


class TeamState:
    def __init__(self, robot_id, fingerprint, *, start_delay_s=.75):
        if robot_id not in {"a", "b"} or start_delay_s < .3:
            raise ValueError("invalid team configuration")
        self.robot_id = robot_id
        self.fingerprint = fingerprint
        self.delay = start_delay_s
        self.phase = "zero"
        self.epoch = 0
        self.proposal = None
        self.ack = 0
        self.committed = 0
        self.event_id = 0
        self.event = ""
        self._seen = {"a": 0, "b": 0}
        self.fault = ""
        self.started_at = 0.

    def request(self, event):
        if event not in {"start", "standing", "task", "stop"}:
            return
        if event == "stop":
            self.fault = "operator_stop"
        self.event_id += 1
        self.event = event

    def status(self, *, ready, frame):
        return dict(robot_id=self.robot_id, fingerprint=self.fingerprint,
                    phase=self.phase, epoch=self.epoch, proposal=self.proposal,
                    ack=self.ack, committed=self.committed, event_id=self.event_id,
                    event=self.event, ready=bool(ready), frame=int(frame), fault=self.fault)

    def update(self, peer, now, *, ready, complete=False):
        """Return True exactly once per locally activated phase transition.

        PREPARE -> ACK -> COMMIT -> commit ACK precedes the future activation
        time. Lost packets are retried by publishing status every tick. This
        cannot guarantee simultaneous action under arbitrary partitions; the
        caller additionally enforces heartbeat, frame and input deadlines.
        """
        other = "b" if self.robot_id == "a" else "a"
        if peer.get("robot_id") != other or peer.get("fingerprint") != self.fingerprint:
            raise RuntimeError("team_identity_or_configuration_mismatch")
        if self.fault or peer.get("fault"):
            raise RuntimeError(self.fault or f"peer_fault: {peer['fault']}")
        peer_epoch = int(peer["epoch"])
        if abs(peer_epoch - self.epoch) > 1:
            raise RuntimeError("team_epoch_skew")
        if peer_epoch < self.epoch and now-self.started_at > .2:
            raise RuntimeError("peer_phase_transition_timeout")
        if peer_epoch == self.epoch and peer["phase"] != self.phase:
            raise RuntimeError("team_phase_mismatch")
        target = None
        for side, event_id, event in (
            (self.robot_id, self.event_id, self.event),
            (other, int(peer["event_id"]), peer["event"]),
        ):
            if event_id > self._seen[side]:
                self._seen[side] = event_id
                if ready and peer["ready"] and peer_epoch == self.epoch:
                    target = {("zero", "start"): "default",
                              ("default", "standing"): "standing",
                              ("standing", "task"): "executing"}.get((self.phase, event)) or target
        if self.robot_id == "a":
            if complete and peer.get("ready") and self.phase == "executing":
                target = "finished"
            if target and self.proposal is None:
                self.proposal = dict(epoch=self.epoch+1, phase=target,
                                     at=now+self.delay, commit=False)
            if self.proposal and int(peer["ack"]) == self.proposal["epoch"]:
                self.proposal["commit"] = True
        else:
            proposal = peer.get("proposal")
            if proposal and int(proposal["epoch"]) > self.epoch:
                if int(proposal["epoch"]) != self.epoch+1:
                    raise RuntimeError("unexpected_transition_epoch")
                valid = {"zero": "default", "default": "standing",
                         "standing": "executing", "executing": "finished"}
                if proposal["phase"] != valid.get(self.phase) or not math.isfinite(proposal["at"]):
                    raise RuntimeError("invalid_team_transition")
                if self.proposal is None and proposal["at"] - now < .1:
                    raise RuntimeError("late_team_transition")
                if self.proposal and any(proposal[k] != self.proposal[k] for k in ("epoch", "phase", "at")):
                    raise RuntimeError("team_transition_changed")
                if not ready:
                    raise RuntimeError("transition_before_local_ready")
                self.proposal = dict(proposal)
                self.ack = self.proposal["epoch"]
                if self.proposal["commit"]:
                    self.committed = self.ack
        if self.proposal and now >= self.proposal["at"]:
            proposal = self.proposal
            if not proposal["commit"] or (self.robot_id == "a" and int(peer["committed"]) != proposal["epoch"]):
                raise RuntimeError("team_transition_not_acknowledged")
            if not ready or (not peer["ready"] and peer_epoch < proposal["epoch"]):
                raise RuntimeError("team_not_ready_at_transition")
            if now - proposal["at"] > .05:
                raise RuntimeError("missed_team_transition_deadline")
            self.phase = proposal["phase"]
            self.epoch = proposal["epoch"]
            self.started_at = proposal["at"]
            self.proposal = None
            return True
        return False
