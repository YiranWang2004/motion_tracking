#!/usr/bin/env bash
set -euo pipefail

# This script only prints the required point-to-point setup by default.
# Applying addresses is intentionally explicit because interface names and
# existing routes are installation-specific.
IFACE_A="${G1_IFACE_A:-enp10s0}"
IFACE_B="${G1_IFACE_B:-enp11s0}"
VETH_A_HOST="${G1_VETH_A_HOST:-10.201.1.1/24}"
VETH_A_NS="${G1_VETH_A_NS:-10.201.1.2/24}"
VETH_B_HOST="${G1_VETH_B_HOST:-10.201.2.1/24}"
VETH_B_NS="${G1_VETH_B_NS:-10.201.2.2/24}"
ROBOT_NET_A="${G1_ROBOT_NET_A:-192.168.123.201/24}"
ROBOT_NET_B="${G1_ROBOT_NET_B:-192.168.123.201/24}"

cat <<EOF
Expected physical wiring:
  ${IFACE_A} -> G1 A (192.168.123.164)
  ${IFACE_B} -> G1 B (192.168.123.164)

Because both robots share the same IP, do not apply this in one namespace.
Use two network namespaces, or re-address one robot first.  A namespace
setup must move ${IFACE_A}/${IFACE_B} into separate namespaces and run the
matching bridge there. Coordinator-to-bridge UDP uses veth pairs, not loopback.

Suggested namespace commands (review before running):
  sudo ip netns add g1a
  sudo ip netns add g1b
  sudo ip link set ${IFACE_A} netns g1a
  sudo ip link set ${IFACE_B} netns g1b
  sudo ip netns exec g1a ip addr add ${ROBOT_NET_A} dev ${IFACE_A}
  sudo ip netns exec g1b ip addr add ${ROBOT_NET_B} dev ${IFACE_B}
  sudo ip netns exec g1a ip link set ${IFACE_A} up
  sudo ip netns exec g1b ip link set ${IFACE_B} up
  sudo ip netns exec g1a ip link set lo up
  sudo ip netns exec g1b ip link set lo up
  sudo ip link add veth-g1a type veth peer name veth-g1a-ns
  sudo ip link add veth-g1b type veth peer name veth-g1b-ns
  sudo ip link set veth-g1a-ns netns g1a
  sudo ip link set veth-g1b-ns netns g1b
  sudo ip addr add ${VETH_A_HOST} dev veth-g1a
  sudo ip addr add ${VETH_B_HOST} dev veth-g1b
  sudo ip netns exec g1a ip addr add ${VETH_A_NS} dev veth-g1a-ns
  sudo ip netns exec g1b ip addr add ${VETH_B_NS} dev veth-g1b-ns
  sudo ip link set veth-g1a up
  sudo ip link set veth-g1b up
  sudo ip netns exec g1a ip link set veth-g1a-ns up
  sudo ip netns exec g1b ip link set veth-g1b-ns up
EOF

if [[ "${1:-}" != "--apply" ]]; then
  exit 0
fi
echo "Refusing --apply: namespace creation requires a site-specific veth/DDS plan." >&2
exit 2
