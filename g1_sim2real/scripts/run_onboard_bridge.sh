#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G1_NET="${G1_NET:-eth0}"
export G1_BRIDGE_CONFIG="${G1_BRIDGE_CONFIG:-${SCRIPT_DIR}/../config/bridge_onboard_scalebfm.yaml}"
exec bash "${SCRIPT_DIR}/run_bridge.sh" "$@"
