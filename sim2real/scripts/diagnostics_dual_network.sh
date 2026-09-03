#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/dual_network_common.sh"

load_dual_network_side a
IFACE_A="${G1_IFACE_A:-${G1_NET_A:-${DUAL_INTERFACE}}}"
NETNS_A="${G1_NETNS_A:-${DUAL_NAMESPACE}}"
ROBOT_IP_A="${DUAL_ROBOT_IP}"
VETH_HOST_A="${DUAL_VETH_HOST_NAME}"

load_dual_network_side b
IFACE_B="${G1_IFACE_B:-${G1_NET_B:-${DUAL_INTERFACE}}}"
NETNS_B="${G1_NETNS_B:-${DUAL_NAMESPACE}}"
ROBOT_IP_B="${DUAL_ROBOT_IP}"
VETH_HOST_B="${DUAL_VETH_HOST_NAME}"

sudo -v

echo "== validated config =="
run_dual_network_reader \
  --config "${DUAL_NETWORK_CONFIG:-${SIM2REAL_ROOT}/config/g1/dual_network.yaml}"
echo "== host veth links and routes =="
ip -br addr show dev "${VETH_HOST_A}" 2>/dev/null || echo "${VETH_HOST_A}: missing"
ip -br addr show dev "${VETH_HOST_B}" 2>/dev/null || echo "${VETH_HOST_B}: missing"
ip route show
echo "== bridge UDP listeners =="
ss -lunp | rg ':(55001|55002|55003|55101|55102|55103)\b' || true
echo "== namespace physical links and robot reachability =="
for entry in "${NETNS_A}:${IFACE_A}:${ROBOT_IP_A}" "${NETNS_B}:${IFACE_B}:${ROBOT_IP_B}"; do
  IFS=: read -r namespace interface robot_ip <<<"${entry}"
  if ip netns list | awk '{print $1}' | grep -Fxq "${namespace}"; then
    echo "[${namespace}]"
    sudo ip -n "${namespace}" -br addr
    sudo ip netns exec "${namespace}" ping -c 1 -W 1 -I "${interface}" "${robot_ip}" || true
    sudo ip -n "${namespace}" neigh show dev "${interface}" || true
    sudo ip netns exec "${namespace}" ss -lunp | rg ':(55001|55002|55003|55101|55102|55103)\b' || true
  else
    echo "[${namespace}] missing"
  fi
done
echo "== optional continuous reachability commands =="
echo "  sudo ip netns exec ${NETNS_A} ping -I ${IFACE_A} ${ROBOT_IP_A}"
echo "  sudo ip netns exec ${NETNS_B} ping -I ${IFACE_B} ${ROBOT_IP_B}"
echo "== optional DDS/RTPS capture commands =="
echo "  sudo ip netns exec ${NETNS_A} tcpdump -ni ${IFACE_A} udp"
echo "  sudo ip netns exec ${NETNS_B} tcpdump -ni ${IFACE_B} udp"
