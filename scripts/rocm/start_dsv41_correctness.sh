#!/usr/bin/env bash
# Reproducible eager V4.1 bring-up. Does not kill an existing service or claim
# readiness/correctness merely because tmux accepted the startup command.
set -euo pipefail
ROOT_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/media/PM983/deepseek-v4.1-flash}"
export HOST="${HOST:-0.0.0.0}" PORT="${PORT:-30101}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TP_SIZE=8 EP_SIZE=1 MOE_A2A_BACKEND=none
export MOE_RUNNER_BACKEND=aiter ATTENTION_BACKEND=dsv4 SGLANG_USE_AITER=1
export TOOL_CALL_PARSER=deepseekv41 REASONING_PARSER=deepseek-v41
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}" MEM_FRACTION_STATIC=0.96
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2304}"
export CUDA_GRAPH_MAX_BS_DECODE=1 DISABLE_DECODE_CUDA_GRAPH=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=1
export SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE=0
export SGLANG_DEBUG_MODEL_STORAGE_AUDIT=1 SGLANG_DSV41_ENGRAM_HOST_CPU_REFERENCE=0
export NO_PROXY=127.0.0.1,localhost,0.0.0.0 no_proxy=127.0.0.1,localhost,0.0.0.0

cd "$ROOT_DIR"
"$PYTHON_BIN" - <<'PY'
import os, socket
with socket.socket() as s:
    s.bind((os.environ["HOST"], int(os.environ["PORT"])))
PY
amd-smi process --gpu 0
MODEL_DIR="$MODEL_PATH" bash scripts/rocm/check_dsv41_consistency.sh
echo "[dsv41-start] Validation passed; starting in foreground. Readiness and HTTP correctness still pending."
exec bash scripts/rocm_dsv4_flash.sh serve
