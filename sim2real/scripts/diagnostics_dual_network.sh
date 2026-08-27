#!/usr/bin/env bash
set -euo pipefail

IFACE_A="${G1_IFACE_A:-enp10s0}"
IFACE_B="${G1_IFACE_B:-enp11s0}"
echo "== host links and routes =="
ip -br addr show
ip route show
echo "== bridge UDP listeners =="
ss -lunp | rg ':(55001|55002|55101|55102)\b' || true
echo "== namespace links =="
for ns in "${G1_NETNS_A:-g1a}" "${G1_NETNS_B:-g1b}"; do
  if ip netns list | awk '{print $1}' | grep -Fxq "${ns}"; then
    echo "[$ns]"
    ip netns exec "${ns}" ip -br addr
    ip netns exec "${ns}" ss -lunp | rg ':(55001|55002|55101|55102)\b' || true
  else
    echo "[$ns] missing"
  fi
done
echo "== DDS/RTPS packets (5 seconds, Ctrl-C to stop) =="
echo "Run separately:"
echo "  sudo ip netns exec ${G1_NETNS_A:-g1a} tcpdump -ni ${IFACE_A} udp"
echo "  sudo ip netns exec ${G1_NETNS_B:-g1b} tcpdump -ni ${IFACE_B} udp"
