#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SIM2REAL_ROOT}"
exec uv run python src/deploy_dual_omnicontact.py "$@"
