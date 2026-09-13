#!/usr/bin/env bash
# Accepted original-V4 TP8 native-AR profile; refuses occupied GPUs via helper.
# Supply a fresh label so previous evidence is never overwritten.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/pc/anaconda3/envs/DS/bin/python \
  "${task_dir}/start-empty-tiles-arm.py" "${1:?Provide a new service label}" \
  --enabled 1 --profile-default --woa-gemv 1
