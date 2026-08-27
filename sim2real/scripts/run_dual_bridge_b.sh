#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../g1_sim2real" && pwd)"
export G1_NET="${G1_NET_B:-enp11s0}"
export G1_BRIDGE_CONFIG="${G1_BRIDGE_CONFIG_B:-${BRIDGE_ROOT}/config/bridge_omnicontact_b.yaml}"
NETNS="${G1_NETNS_B:-g1b}"
if ! ip netns list | awk '{print $1}' | grep -Fxq "${NETNS}"; then
  echo "Network namespace not found: ${NETNS}. Run setup_dual_network.sh first." >&2
  exit 1
fi
exec sudo ip netns exec "${NETNS}" env \
  G1_NET="${G1_NET}" G1_BRIDGE_CONFIG="${G1_BRIDGE_CONFIG}" \
  bash "${BRIDGE_ROOT}/scripts/run_bridge.sh" "$@"
