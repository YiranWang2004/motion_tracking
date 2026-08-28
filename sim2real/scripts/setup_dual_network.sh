#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/dual_network_common.sh"

usage() {
  cat <<'EOF'
Usage: setup_dual_network.sh [--apply|--check|--teardown]

  no option  Validate config and print the topology without changing the host.
  --apply    Idempotently create namespaces/veths, move physical NICs, and set IPs.
  --check    Verify that the configured topology is already present.
  --teardown Stop using the topology, return NICs to the host, and delete namespaces.

Set DUAL_NETWORK_CONFIG only to select a non-default YAML file.
EOF
}

MODE="preview"
case "${1:-}" in
  "") ;;
  --apply) MODE="apply" ;;
  --check) MODE="check" ;;
  --teardown) MODE="teardown" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

load_dual_network_side a
IFACE_A="${G1_IFACE_A:-${G1_NET_A:-${DUAL_INTERFACE}}}"
EXPECTED_MAC_A="${DUAL_EXPECTED_MAC}"
NETNS_A="${G1_NETNS_A:-${DUAL_NAMESPACE}}"
ROBOT_ADDRESS_A="${G1_ROBOT_NET_A:-${DUAL_ROBOT_ADDRESS}}"
ROBOT_IP_A="${DUAL_ROBOT_IP}"
VETH_HOST_A="${G1_VETH_A_HOST_NAME:-${DUAL_VETH_HOST_NAME}}"
VETH_HOST_ADDRESS_A="${G1_VETH_A_HOST:-${DUAL_VETH_HOST_ADDRESS}}"
VETH_NS_A="${G1_VETH_A_NS_NAME:-${DUAL_VETH_NAMESPACE_NAME}}"
VETH_NS_ADDRESS_A="${G1_VETH_A_NS:-${DUAL_VETH_NAMESPACE_ADDRESS}}"

load_dual_network_side b
IFACE_B="${G1_IFACE_B:-${G1_NET_B:-${DUAL_INTERFACE}}}"
EXPECTED_MAC_B="${DUAL_EXPECTED_MAC}"
NETNS_B="${G1_NETNS_B:-${DUAL_NAMESPACE}}"
ROBOT_ADDRESS_B="${G1_ROBOT_NET_B:-${DUAL_ROBOT_ADDRESS}}"
ROBOT_IP_B="${DUAL_ROBOT_IP}"
VETH_HOST_B="${G1_VETH_B_HOST_NAME:-${DUAL_VETH_HOST_NAME}}"
VETH_HOST_ADDRESS_B="${G1_VETH_B_HOST:-${DUAL_VETH_HOST_ADDRESS}}"
VETH_NS_B="${G1_VETH_B_NS_NAME:-${DUAL_VETH_NAMESPACE_NAME}}"
VETH_NS_ADDRESS_B="${G1_VETH_B_NS:-${DUAL_VETH_NAMESPACE_ADDRESS}}"

print_topology() {
  cat <<EOF
Dual G1 network config: ${DUAL_NETWORK_CONFIG:-${SIM2REAL_ROOT}/config/g1/dual_network.yaml}

  G1 A: ${IFACE_A} (${EXPECTED_MAC_A}) -> namespace ${NETNS_A} -> ${ROBOT_IP_A}
        physical=${ROBOT_ADDRESS_A}
        host ${VETH_HOST_A}=${VETH_HOST_ADDRESS_A} <-> ${VETH_NS_A}=${VETH_NS_ADDRESS_A}
  G1 B: ${IFACE_B} (${EXPECTED_MAC_B}) -> namespace ${NETNS_B} -> ${ROBOT_IP_B}
        physical=${ROBOT_ADDRESS_B}
        host ${VETH_HOST_B}=${VETH_HOST_ADDRESS_B} <-> ${VETH_NS_B}=${VETH_NS_ADDRESS_B}
EOF
}

namespace_exists() {
  ip netns list | awk '{print $1}' | grep -Fxq "$1"
}

namespace_link_exists() {
  ip -n "$1" link show dev "$2" >/dev/null 2>&1
}

address_present() {
  local namespace="$1"
  local interface="$2"
  local address="$3"
  if [[ -n "${namespace}" ]]; then
    ip -n "${namespace}" -o addr show dev "${interface}" | awk '{print $4}' | grep -Fxq "${address}"
  else
    ip -o addr show dev "${interface}" | awk '{print $4}' | grep -Fxq "${address}"
  fi
}

link_mac() {
  local namespace="$1"
  local interface="$2"
  if [[ -n "${namespace}" ]]; then
    ip -n "${namespace}" -o link show dev "${interface}"
  else
    ip -o link show dev "${interface}"
  fi | awk '{for (i=1; i<=NF; i++) if ($i == "link/ether") print $(i+1)}'
}

preflight_side() {
  local label="$1"
  local namespace="$2"
  local physical="$3"
  local expected_mac="$4"
  local veth_host="$5"
  local veth_ns="$6"
  local location=""
  if ip link show dev "${physical}" >/dev/null 2>&1; then
    location=""
  elif namespace_exists "${namespace}" && namespace_link_exists "${namespace}" "${physical}"; then
    location="${namespace}"
  else
    echo "${label}: physical interface ${physical} is missing from host and ${namespace}" >&2
    return 1
  fi
  local actual_mac
  actual_mac="$(link_mac "${location}" "${physical}")"
  if [[ "${actual_mac,,}" != "${expected_mac,,}" ]]; then
    echo "${label}: ${physical} MAC ${actual_mac:-missing} does not match configured ${expected_mac}" >&2
    return 1
  fi
  local host_exists=0
  local ns_exists=0
  ip link show dev "${veth_host}" >/dev/null 2>&1 && host_exists=1
  if namespace_exists "${namespace}" && namespace_link_exists "${namespace}" "${veth_ns}"; then
    ns_exists=1
  fi
  if [[ "${host_exists}" -ne "${ns_exists}" ]]; then
    echo "${label}: partial veth exists; refusing to modify the topology" >&2
    return 1
  fi
}

check_side() {
  local label="$1"
  local namespace="$2"
  local physical="$3"
  local physical_address="$4"
  local veth_host="$5"
  local veth_host_address="$6"
  local veth_ns="$7"
  local veth_ns_address="$8"
  local robot_ip="$9"
  local expected_mac="${10}"
  local failed=0
  namespace_exists "${namespace}" || { echo "${label}: missing namespace ${namespace}" >&2; failed=1; }
  if namespace_exists "${namespace}"; then
    namespace_link_exists "${namespace}" "${physical}" || { echo "${label}: ${physical} is not in ${namespace}" >&2; failed=1; }
    namespace_link_exists "${namespace}" "${veth_ns}" || { echo "${label}: missing ${namespace}/${veth_ns}" >&2; failed=1; }
    if namespace_link_exists "${namespace}" "${physical}"; then
      address_present "${namespace}" "${physical}" "${physical_address}" || { echo "${label}: missing ${physical_address} on ${physical}" >&2; failed=1; }
      local actual_mac
      actual_mac="$(link_mac "${namespace}" "${physical}")"
      [[ "${actual_mac,,}" == "${expected_mac,,}" ]] || { echo "${label}: ${physical} MAC ${actual_mac:-missing} does not match ${expected_mac}" >&2; failed=1; }
    fi
    if namespace_link_exists "${namespace}" "${veth_ns}"; then
      address_present "${namespace}" "${veth_ns}" "${veth_ns_address}" || { echo "${label}: missing ${veth_ns_address} on ${veth_ns}" >&2; failed=1; }
    fi
  fi
  ip link show dev "${veth_host}" >/dev/null 2>&1 || { echo "${label}: missing host ${veth_host}" >&2; failed=1; }
  if ip link show dev "${veth_host}" >/dev/null 2>&1; then
    address_present "" "${veth_host}" "${veth_host_address}" || { echo "${label}: missing ${veth_host_address} on ${veth_host}" >&2; failed=1; }
  fi
  if [[ "${failed}" -eq 0 ]]; then
    echo "${label}: topology OK (${physical} -> ${robot_ip})"
  fi
  return "${failed}"
}

ensure_side() {
  local label="$1"
  local namespace="$2"
  local physical="$3"
  local physical_address="$4"
  local veth_host="$5"
  local veth_host_address="$6"
  local veth_ns="$7"
  local veth_ns_address="$8"

  if ! namespace_exists "${namespace}"; then
    ip netns add "${namespace}"
  fi
  if ip link show dev "${physical}" >/dev/null 2>&1; then
    ip link set "${physical}" netns "${namespace}"
  elif ! namespace_link_exists "${namespace}" "${physical}"; then
    echo "${label}: physical interface ${physical} is missing from host and ${namespace}" >&2
    return 1
  fi

  local host_exists=0
  local ns_exists=0
  ip link show dev "${veth_host}" >/dev/null 2>&1 && host_exists=1
  namespace_link_exists "${namespace}" "${veth_ns}" && ns_exists=1
  if [[ "${host_exists}" -eq 0 && "${ns_exists}" -eq 0 ]]; then
    ip link add "${veth_host}" type veth peer name "${veth_ns}"
    ip link set "${veth_ns}" netns "${namespace}"
  elif [[ "${host_exists}" -ne 1 || "${ns_exists}" -ne 1 ]]; then
    echo "${label}: partial veth exists; refusing to delete or replace it" >&2
    return 1
  fi

  ip -n "${namespace}" addr replace "${physical_address}" dev "${physical}"
  ip -n "${namespace}" addr replace "${veth_ns_address}" dev "${veth_ns}"
  ip addr replace "${veth_host_address}" dev "${veth_host}"
  ip -n "${namespace}" link set lo up
  ip -n "${namespace}" link set "${physical}" up
  ip -n "${namespace}" link set "${veth_ns}" up
  ip link set "${veth_host}" up
}

teardown_side() {
  local label="$1"
  local namespace="$2"
  local physical="$3"
  local veth_host="$4"
  if ip link show dev "${veth_host}" >/dev/null 2>&1; then
    ip link del "${veth_host}"
  fi
  if namespace_exists "${namespace}"; then
    if namespace_link_exists "${namespace}" "${physical}"; then
      ip -n "${namespace}" link set "${physical}" netns 1
    else
      echo "${label}: ${physical} was not in ${namespace}; leaving interfaces untouched" >&2
    fi
    ip netns del "${namespace}"
  fi
  if ip link show dev "${physical}" >/dev/null 2>&1; then
    ip link set "${physical}" up
  fi
}

print_topology
if [[ "${MODE}" == "preview" ]]; then
  echo
  echo "Preview only. Apply once per boot with:"
  echo "  sudo bash scripts/setup_dual_network.sh --apply"
  exit 0
fi

if [[ "${MODE}" == "apply" || "${MODE}" == "teardown" ]]; then
  if [[ "${EUID}" -ne 0 ]]; then
    echo "${MODE} requires root; run with sudo" >&2
    exit 1
  fi
fi

if [[ "${MODE}" == "teardown" ]]; then
  teardown_side "G1 A" "${NETNS_A}" "${IFACE_A}" "${VETH_HOST_A}"
  teardown_side "G1 B" "${NETNS_B}" "${IFACE_B}" "${VETH_HOST_B}"
  echo "dual G1 namespaces removed; restore any site NetworkManager profiles if needed"
  exit 0
fi

if [[ "${MODE}" == "apply" ]]; then
  preflight_side "G1 A" "${NETNS_A}" "${IFACE_A}" "${EXPECTED_MAC_A}" "${VETH_HOST_A}" "${VETH_NS_A}"
  preflight_side "G1 B" "${NETNS_B}" "${IFACE_B}" "${EXPECTED_MAC_B}" "${VETH_HOST_B}" "${VETH_NS_B}"
  ensure_side "G1 A" "${NETNS_A}" "${IFACE_A}" "${ROBOT_ADDRESS_A}" \
    "${VETH_HOST_A}" "${VETH_HOST_ADDRESS_A}" "${VETH_NS_A}" "${VETH_NS_ADDRESS_A}"
  ensure_side "G1 B" "${NETNS_B}" "${IFACE_B}" "${ROBOT_ADDRESS_B}" \
    "${VETH_HOST_B}" "${VETH_HOST_ADDRESS_B}" "${VETH_NS_B}" "${VETH_NS_ADDRESS_B}"
fi

failed=0
check_side "G1 A" "${NETNS_A}" "${IFACE_A}" "${ROBOT_ADDRESS_A}" \
  "${VETH_HOST_A}" "${VETH_HOST_ADDRESS_A}" "${VETH_NS_A}" "${VETH_NS_ADDRESS_A}" "${ROBOT_IP_A}" "${EXPECTED_MAC_A}" || failed=1
check_side "G1 B" "${NETNS_B}" "${IFACE_B}" "${ROBOT_ADDRESS_B}" \
  "${VETH_HOST_B}" "${VETH_HOST_ADDRESS_B}" "${VETH_NS_B}" "${VETH_NS_ADDRESS_B}" "${ROBOT_IP_B}" "${EXPECTED_MAC_B}" || failed=1
exit "${failed}"
