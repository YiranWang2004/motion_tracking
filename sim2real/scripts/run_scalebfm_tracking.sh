#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Never implicitly sync/change a configured robot Python environment.
exec "${SCALEBFM_PYTHON:-${PROJECT_DIR}/.venv/bin/python}" -m scalebfm_tracking.runner "$@"
