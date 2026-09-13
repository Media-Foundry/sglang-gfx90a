#!/usr/bin/env bash
# CPU contract tests with the same local ROCm AOT import paths as the launcher.
# No checkpoint loading, service restart, GPU benchmark or HTTP requests.
set -euo pipefail

ROOT_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:${ROOT_DIR}/python/sglang/kernels/aot/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_USE_AITER=1
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m pytest -q \
  test/registered/unit/test_deepseek_v41_bringup.py \
  test/registered/unit/entrypoints/openai/test_dsv41_chat_encoding_contract.py \
  test/registered/unit/function_call/test_deepseekv41_detector.py \
  test/registered/unit/layers/attention/test_dsv41_candidate_blocks.py \
  test/registered/unit/layers/attention/test_dspark_swa_loc_replay.py \
  test/registered/unit/layers/attention/test_dsv41_bounded_scores.py \
  test/registered/unit/test_dsv41_mixed_context_harness.py "$@"
