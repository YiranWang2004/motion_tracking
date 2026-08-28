#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo -v
sudo bash "${SCRIPT_DIR}/setup_dual_network.sh" --check
bash "${SCRIPT_DIR}/run_dual_bridge_a.sh" "$@" &
PID_A=$!
bash "${SCRIPT_DIR}/run_dual_bridge_b.sh" "$@" &
PID_B=$!

cleanup() {
  trap - INT TERM EXIT
  kill "${PID_A}" "${PID_B}" 2>/dev/null || true
  wait "${PID_A}" "${PID_B}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "dual bridges started: A pid=${PID_A}, B pid=${PID_B}"
echo "Stopping either bridge stops the pair. Press Ctrl-C to stop both."
set +e
wait -n "${PID_A}" "${PID_B}"
status=$?
set -e
echo "one bridge exited with status ${status}; stopping the other" >&2
exit "${status}"
