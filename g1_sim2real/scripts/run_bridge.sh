#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
BUILD_DIR="${G1_BRIDGE_BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/g1_udp_bridge"
CONFIG="${G1_BRIDGE_CONFIG:-${ROOT_DIR}/config/g1_bridge.yaml}"
NET="${G1_NET:-lo}"

if [[ ! -x "${BIN}" ]]; then
  echo "Bridge binary not found: ${BIN}" >&2
  echo "Run: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

if [[ "${G1_BRIDGE_NO_FILE_LOG:-0}" == "1" ]]; then
  exec "${BIN}" --net "${NET}" --config "${CONFIG}" "$@"
fi

LOG_DIR="${G1_BRIDGE_LOG_DIR:-${PROJECT_DIR}/logs/omnicontact}"
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${G1_BRIDGE_LOG_FILE:-${LOG_DIR}/bridge_${LOG_STAMP}_pid$$.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "[G1BridgeLauncher] session log file: ${LOG_FILE}"
{
  echo "[G1BridgeLauncher] session_start=$(date --iso-8601=seconds) pid=$$"
  echo "[G1BridgeLauncher] binary=${BIN}"
  echo "[G1BridgeLauncher] config=${CONFIG}"
  echo "[G1BridgeLauncher] network_interface=${NET}"
  set +e
  "${BIN}" --net "${NET}" --config "${CONFIG}" "$@"
  bridge_status=$?
  set -e
  echo "[G1BridgeLauncher] session_end=$(date --iso-8601=seconds) status=${bridge_status}"
  exit "${bridge_status}"
} 2>&1 | tee -a "${LOG_FILE}"
