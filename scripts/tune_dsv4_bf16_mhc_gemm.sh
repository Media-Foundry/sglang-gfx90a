#!/usr/bin/env bash
set -euo pipefail

AITer_DIR="${AITER_DIR:-/home/pc/pytorch/third_party/aiter}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
OUT_DIR="${OUT_DIR:-/tmp/aiter_configs}"
IN_FILE="${IN_FILE:-${OUT_DIR}/dsv4_bf16_mhc_untuned_gemm.csv}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/dsv4_bf16_mhc_tuned_gemm.csv}"
PROFILE_FILE="${PROFILE_FILE:-${OUT_DIR}/dsv4_bf16_mhc_profile_gemm.csv}"
M_SET="${M_SET:-1,2,4}"
N_SET="${N_SET:-512,1024,2048}"

mkdir -p "${OUT_DIR}"
{
  echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
  IFS=',' read -r -a m_values <<<"${M_SET}"
  IFS=',' read -r -a n_values <<<"${N_SET}"
  for n in "${n_values[@]}"; do
    for m in "${m_values[@]}"; do
      echo "${m},${n},4096,False,torch.bfloat16,torch.bfloat16,False,False"
    done
  done
} >"${IN_FILE}"

echo "AITer BF16 GEMM tuning input: ${IN_FILE}"
cat "${IN_FILE}"
cat <<EOF
Output: ${OUT_FILE}
Profile: ${PROFILE_FILE}
LIBTYPE=${LIBTYPE:-all} MP=${MP:-1} BATCH=${BATCH:-1} WARMUP=${WARMUP:-10} ITERS=${ITERS:-50} TIMEOUT=${TIMEOUT:-240}
EOF

: <<'CSV'
M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle
1,512,4096,False,torch.bfloat16,torch.bfloat16,False,False
2,512,4096,False,torch.bfloat16,torch.bfloat16,False,False
4,512,4096,False,torch.bfloat16,torch.bfloat16,False,False
1,1024,4096,False,torch.bfloat16,torch.bfloat16,False,False
2,1024,4096,False,torch.bfloat16,torch.bfloat16,False,False
4,1024,4096,False,torch.bfloat16,torch.bfloat16,False,False
1,2048,4096,False,torch.bfloat16,torch.bfloat16,False,False
2,2048,4096,False,torch.bfloat16,torch.bfloat16,False,False
4,2048,4096,False,torch.bfloat16,torch.bfloat16,False,False
CSV

cd "${AITer_DIR}"
exec "${PYTHON_BIN}" gradlib/gradlib/gemm_tuner.py \
  --input_file "${IN_FILE}" \
  --tuned_file "${OUT_FILE}" \
  --profile_file "${PROFILE_FILE}" \
  --mp "${MP:-1}" \
  --batch "${BATCH:-1}" \
  --warmup "${WARMUP:-10}" \
  --iters "${ITERS:-50}" \
  --timeout "${TIMEOUT:-240}" \
  --libtype "${LIBTYPE:-all}" \
  --sort \
  --verbose
