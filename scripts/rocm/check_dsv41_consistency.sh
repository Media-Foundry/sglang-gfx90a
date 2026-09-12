#!/usr/bin/env bash
# Run the cheap, repeatable V4.1 correctness/storage checks before a service
# startup or benchmark.  This script does not load model weights onto GPUs.
set -euo pipefail

ROOT_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
MODEL_DIR="${MODEL_DIR:-/media/PM983/deepseek-v4.1-flash}"
TP_SIZE="${TP_SIZE:-8}"
LOG_FILE="${1:-}"

cd "${ROOT_DIR}"
echo "[dsv41-check] repo=$(git rev-parse --show-toplevel)"
echo "[dsv41-check] commit=$(git rev-parse --short HEAD)"
echo "[dsv41-check] model=${MODEL_DIR} tp=${TP_SIZE}"

"${PYTHON_BIN}" scripts/rocm/audit_deepseek_v41_checkpoint.py \
  --model-dir "${MODEL_DIR}" --tp-size "${TP_SIZE}" \
  --require-complete --require-engram \
  --json-out /tmp/dsv41_checkpoint_audit.json

# This probes only a few Engram rows and static headers; it does not allocate
# the full host tables or any GPU model storage.
"${PYTHON_BIN}" scripts/rocm/smoke_deepseek_v41_engram.py \
  --model-dir "${MODEL_DIR}" --tp-size "${TP_SIZE}" --rows 3

git diff --check
"${PYTHON_BIN}" -m py_compile \
  python/sglang/srt/layers/engram.py \
  python/sglang/srt/layers/quantization/fp8.py \
  python/sglang/srt/layers/moe/moe_runner/aiter.py \
  python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py \
  python/sglang/srt/models/deepseek_v41.py
bash -n scripts/rocm_dsv4_flash.sh

if [[ -n "${LOG_FILE}" ]]; then
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "[dsv41-check] log not found: ${LOG_FILE}" >&2
    exit 2
  fi
  echo "[dsv41-check] storage lines from ${LOG_FILE}"
  rg -n "model storage audit|engram host table|Memory pool end|HSA_STATUS_ERROR_MEMORY_FAULT|_engram_gather_kernel|cudaHostRegister" "${LOG_FILE}" || true
fi

echo "[dsv41-check] PASS: checkpoint, Engram header/row probe, source syntax"
