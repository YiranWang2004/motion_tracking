#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use desktop Python: PyYAML and ConfigObj are provided by the host packages.
exec /usr/bin/python3 "${SCRIPT_DIR}/open_onboard_terminals.py" "$@"
