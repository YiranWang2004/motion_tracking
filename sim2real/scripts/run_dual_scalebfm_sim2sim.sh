#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SIM2REAL_ROOT}"

sim_args=()
deploy_args=()
while (($#)); do
  case "$1" in
    --headless)
      sim_args+=("$1")
      ;;
    --sim-max-policy-steps)
      [[ $# -ge 2 ]] || { echo "--sim-max-policy-steps requires a value" >&2; exit 2; }
      sim_args+=(--max-policy-steps "$2")
      shift
      ;;
    *)
      deploy_args+=("$1")
      ;;
  esac
  shift
done

uv run python src/dual_scalebfm_sim2sim.py "${sim_args[@]}" &
sim_pid=$!
cleanup() {
  kill "${sim_pid}" 2>/dev/null || true
  wait "${sim_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uv run --extra dual-policy python src/deploy_dual_scalebfm_residual.py \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization \
  "${deploy_args[@]}"
