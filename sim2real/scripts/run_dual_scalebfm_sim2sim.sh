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
    --config|--reference-bundle)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      sim_args+=("$1" "$2")
      deploy_args+=("$1" "$2")
      shift
      ;;
    *)
      deploy_args+=("$1")
      ;;
  esac
  shift
done

# Background processes otherwise inherit /dev/null as stdin. Preserve terminal
# input for headless s/b/a/x control; a pipeline can also provide scripted keys.
exec 3<&0
uv run python src/dual_scalebfm_sim2sim.py "${sim_args[@]}" <&3 &
sim_pid=$!
deploy_pid=""
cleanup() {
  if [[ -n "${deploy_pid}" ]]; then
    kill "${deploy_pid}" 2>/dev/null || true
    wait "${deploy_pid}" 2>/dev/null || true
  fi
  kill "${sim_pid}" 2>/dev/null || true
  wait "${sim_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uv run --extra dual-policy python src/deploy_dual_scalebfm_residual.py \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization \
  "${deploy_args[@]}" &
deploy_pid=$!

set +e
wait -n "${sim_pid}" "${deploy_pid}"
status=$?
set -e
exit "${status}"
