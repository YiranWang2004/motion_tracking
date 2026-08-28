#!/usr/bin/env bash

# Source-only helper. The Python reader validates YAML, addresses, unique names,
# bridge paths, and bridge/veth endpoint consistency before returning values.
run_dual_network_reader() {
  if command -v uv >/dev/null 2>&1; then
    uv run --project "${SIM2REAL_ROOT}" python \
      "${SIM2REAL_ROOT}/scripts/read_dual_network_config.py" "$@"
  elif /usr/bin/python3 -c 'import yaml' >/dev/null 2>&1; then
    PYTHONPATH="${SIM2REAL_ROOT}/src" /usr/bin/python3 \
      "${SIM2REAL_ROOT}/scripts/read_dual_network_config.py" "$@"
  else
    echo "Neither uv nor a Python 3 installation with PyYAML is available" >&2
    return 1
  fi
}

load_dual_network_side() {
  local side="$1"
  local output
  local config_path="${DUAL_NETWORK_CONFIG:-${SIM2REAL_ROOT}/config/g1/dual_network.yaml}"
  if ! output="$(
    run_dual_network_reader --config "${config_path}" --side "${side}"
  )"; then
    echo "Failed to load dual network config: ${config_path}" >&2
    return 1
  fi
  mapfile -t DUAL_NETWORK_FIELDS <<<"${output}"
  if [[ "${#DUAL_NETWORK_FIELDS[@]}" -ne 11 ]]; then
    echo "Invalid output from dual network config reader" >&2
    return 1
  fi
  DUAL_ROBOT_ID="${DUAL_NETWORK_FIELDS[0]}"
  DUAL_INTERFACE="${DUAL_NETWORK_FIELDS[1]}"
  DUAL_NAMESPACE="${DUAL_NETWORK_FIELDS[2]}"
  DUAL_ROBOT_ADDRESS="${DUAL_NETWORK_FIELDS[3]}"
  DUAL_ROBOT_IP="${DUAL_NETWORK_FIELDS[4]}"
  DUAL_VETH_HOST_NAME="${DUAL_NETWORK_FIELDS[5]}"
  DUAL_VETH_HOST_ADDRESS="${DUAL_NETWORK_FIELDS[6]}"
  DUAL_VETH_NAMESPACE_NAME="${DUAL_NETWORK_FIELDS[7]}"
  DUAL_VETH_NAMESPACE_ADDRESS="${DUAL_NETWORK_FIELDS[8]}"
  DUAL_BRIDGE_CONFIG="${DUAL_NETWORK_FIELDS[9]}"
  DUAL_EXPECTED_MAC="${DUAL_NETWORK_FIELDS[10]}"
}
