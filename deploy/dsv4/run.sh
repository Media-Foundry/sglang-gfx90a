#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "${KIT_DIR}/deploy.py" "$@"
