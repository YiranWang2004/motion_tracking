#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../g1_sim2real" && pwd)"
source "${SCRIPT_DIR}/dual_network_common.sh"
load_dual_network_side a
export G1_NET="${G1_NET_A:-${DUAL_INTERFACE}}"
export G1_BRIDGE_CONFIG="${G1_BRIDGE_CONFIG_A:-${DUAL_BRIDGE_CONFIG}}"
NETNS="${G1_NETNS_A:-${DUAL_NAMESPACE}}"
if ! ip netns list | awk '{print $1}' | grep -Fxq "${NETNS}"; then
  echo "Network namespace not found: ${NETNS}. Run setup_dual_network.sh --apply first." >&2
  exit 1
fi
if ! sudo ip -n "${NETNS}" link show dev "${G1_NET}" >/dev/null 2>&1; then
  echo "Configured interface ${G1_NET} is not inside namespace ${NETNS}." >&2
  exit 1
fi
ACTUAL_MAC="$(sudo ip -n "${NETNS}" -o link show dev "${G1_NET}" | awk '{for (i=1; i<=NF; i++) if ($i == "link/ether") print $(i+1)}')"
if [[ "${ACTUAL_MAC,,}" != "${DUAL_EXPECTED_MAC,,}" ]]; then
  echo "Configured interface ${G1_NET} has MAC ${ACTUAL_MAC}, expected ${DUAL_EXPECTED_MAC}." >&2
  exit 1
fi
exec sudo ip netns exec "${NETNS}" env \
  G1_NET="${G1_NET}" G1_BRIDGE_CONFIG="${G1_BRIDGE_CONFIG}" \
  bash "${BRIDGE_ROOT}/scripts/run_bridge.sh" "$@"
