#!/usr/bin/env bash
set -euo pipefail

AITER_DIR="${AITER_DIR:-/home/pc/pytorch/third_party/aiter}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm/core-7.14}"
OUT_DIR="${OUT_DIR:-/tmp/aiter_configs}"
IN_FILE="${IN_FILE:-${OUT_DIR}/dsv4_bf16_decode_untuned_gemm.csv}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/dsv4_bf16_decode_tuned_gemm.csv}"
PROFILE_FILE="${PROFILE_FILE:-${OUT_DIR}/dsv4_bf16_decode_profile_gemm.csv}"

mkdir -p "${OUT_DIR}"
cat >"${IN_FILE}" <<'CSV'
M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle
1,1536,4096,False,torch.bfloat16,torch.bfloat16,False,False
1,8192,1024,False,torch.bfloat16,torch.bfloat16,False,False
1,4096,2048,False,torch.bfloat16,torch.bfloat16,False,False
1,4096,4096,False,torch.bfloat16,torch.bfloat16,False,False
CSV

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PATH="${ROCM_ROOT}/bin:${PATH}"
export ROCM_PATH="${ROCM_ROOT}"
export CMAKE_PREFIX_PATH="${ROCM_ROOT}:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="${AITER_DIR}:${PYTHONPATH:-}"

cd "${AITER_DIR}"
exec "${PYTHON_BIN}" gradlib/gradlib/gemm_tuner.py \
  --input_file "${IN_FILE}" \
  --tuned_file "${OUT_FILE}" \
  --profile_file "${PROFILE_FILE}" \
  --mp "${MP:-1}" \
  --batch "${BATCH:-1}" \
  --warmup "${WARMUP:-10}" \
  --iters "${ITERS:-40}" \
  --timeout "${TIMEOUT:-300}" \
  --libtype "${LIBTYPE:-all}" \
  --sort \
  --verbose
