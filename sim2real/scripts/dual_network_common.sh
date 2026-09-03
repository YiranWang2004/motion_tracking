#!/usr/bin/env bash

# Source-only helper. The Python reader validates YAML, addresses, unique names,
# bridge paths, and bridge/veth endpoint consistency before returning values.
run_dual_network_reader() {
  local uv_bin=""
  local uv_user=""
  if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
  elif [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    # sudo normally replaces PATH with secure_path, which hides a per-user uv
    # installation. Locate the invoking user's uv and run it as that user so
    # the project environment/cache never becomes root-owned.
    local sudo_user_home=""
    sudo_user_home="$(getent passwd "${SUDO_USER}" | awk -F: 'NR == 1 {print $6}')"
    local candidate
    for candidate in \
      "${sudo_user_home}/.local/bin/uv" \
      "${sudo_user_home}/.cargo/bin/uv"; do
      if [[ -n "${sudo_user_home}" && -x "${candidate}" ]]; then
        uv_bin="${candidate}"
        uv_user="${SUDO_USER}"
        break
      fi
    done
  fi

  if [[ -n "${uv_bin}" && -n "${uv_user}" ]]; then
    sudo -u "${uv_user}" -H "${uv_bin}" run --project "${SIM2REAL_ROOT}" python \
      "${SIM2REAL_ROOT}/scripts/read_dual_network_config.py" "$@"
  elif [[ -n "${uv_bin}" ]]; then
    "${uv_bin}" run --project "${SIM2REAL_ROOT}" python \
      "${SIM2REAL_ROOT}/scripts/read_dual_network_config.py" "$@"
  elif command -v python3 >/dev/null 2>&1 && \
    python3 -c 'import sys, numpy, yaml; raise SystemExit(sys.version_info < (3, 10))' \
      >/dev/null 2>&1; then
    PYTHONPATH="${SIM2REAL_ROOT}/src" python3 \
      "${SIM2REAL_ROOT}/scripts/read_dual_network_config.py" "$@"
  else
    echo "Cannot run the dual-network config reader: uv was not found, and Python >= 3.10 with the project dependencies is unavailable" >&2
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
