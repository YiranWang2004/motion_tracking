"""Transparent, allowlisted pose/team UDP relays; never a G1 low-level bridge.

    host publisher <-> namespace pose relay <-> onboard pose receiver
    onboard A <-> namespace A team relay <-> host hub <-> namespace B <-> onboard B

Clock pings/pongs and timestamps travel unchanged, so each endpoint estimates
the actual remote clock across the complete path, including relay latency.
"""
import json
import select
import socket
import threading

from .onboard_network import MAX_PACKET, PROTOCOL


def allowed_packet(data, kind):
    try:
        if len(data) > MAX_PACKET:
            return False
        msg = json.loads(data)
        if msg.get("protocol") != PROTOCOL:
            return False
        if msg.get("kind") == "ping":
            return set(msg) == {"protocol", "kind", "token", "t0"}
        if msg.get("kind") == "pong":
            return set(msg) == {"protocol", "kind", "token", "t0", "t1", "t2"}
        if msg.get("kind") != kind or set(msg) != {"protocol", "kind", "stream", "seq", "sent", "payload"}:
            return False
        payload = msg["payload"]
        fields = ({"calibration_id", "snapshot_seq", "captured", "poses", "half_extents"}
                  if kind == "pose" else
                  {"robot_id", "fingerprint", "phase", "epoch", "proposal", "ack",
                   "committed", "event_id", "event", "ready", "frame", "fault"})
        if not isinstance(payload, dict) or set(payload) != fields:
            return False
        if kind == "team" and payload["proposal"] is not None:
            if not isinstance(payload["proposal"], dict) or set(payload["proposal"]) != {"epoch", "phase", "at", "commit"}:
                return False
        return True
    except (ValueError, TypeError, AttributeError):
        return False


class DatagramRelay:
    """Two explicit bind/peer legs. No routing/NAT changes or clock termination."""
    def __init__(self, left, right, kind):
        if kind not in {"pose", "team"}:
            raise ValueError("only pose and team traffic may be relayed")
        self.kind = kind
        self.sockets = []
        self.peers = []
        self.forwarded = 0
        self.dropped = 0
        self.error = None
        self.stop_event = threading.Event()
        try:
            for bind, peer in (left, right):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets.append(sock)
                sock.bind(bind)
                sock.setblocking(False)
                self.peers.append((socket.gethostbyname(peer[0]), int(peer[1])))
        except BaseException:
            for sock in self.sockets:
                sock.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"{kind}-relay")
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                readable, _, _ = select.select(self.sockets, [], [], .05)
                for sock in readable:
                    index = self.sockets.index(sock)
                    data, source = sock.recvfrom(MAX_PACKET+1)
                    if source != self.peers[index] or not allowed_packet(data, self.kind):
                        self.dropped += 1
                        continue
                    try:
                        self.sockets[1-index].sendto(data, self.peers[1-index])
                        self.forwarded += 1
                    except BlockingIOError:
                        self.dropped += 1
        except OSError as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=.2)
        for sock in self.sockets:
            sock.close()


def namespace_relays(net, side):
    local = net[side]
    result = []
    try:
        for kind, port in (("pose", local["pose_port"]), ("team", local["team_port"])):
            host_port = local["host_pose_port"] if kind == "pose" else port
            result.append(DatagramRelay(
                ((local["relay_veth_host"], port), (local["publisher_host"], host_port)),
                ((local["relay_host"], host_port), (local["host"], port)), kind))
        return result
    except BaseException:
        for relay in result:
            relay.close()
        raise


def team_hub(net):
    legs = [((net[s]["publisher_host"], net[s]["team_port"]),
             (net[s]["relay_veth_host"], net[s]["team_port"])) for s in ("a", "b")]
    return DatagramRelay(*legs, "team")
