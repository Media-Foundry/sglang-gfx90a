#!/usr/bin/env bash
set -euo pipefail

# Native-AR Qwen3.8Next serving profile for four MI250/gfx90a GCDs.
# BF16 recurrent state is a correctness-verified decode optimization and also
# halves the GDN state-pool footprint.  Set MAMBA_SSM_DTYPE=float32 for the
# reference/rollback arm of an A/B test.

MODEL_PATH=${MODEL_PATH:-/media/PM983/qwen3.8next}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30001}
MAMBA_SSM_DTYPE=${MAMBA_SSM_DTYPE:-bfloat16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.65}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4096}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-64}
CUDA_GRAPH_BS_DECODE=${CUDA_GRAPH_BS_DECODE:-"1 16 32"}
INIT_EXPERT_LOCATION=${INIT_EXPERT_LOCATION:-${SCRIPT_DIR}/rocm/qwen38next_eplb_static_lpt.json}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
# The launcher freezes GC in every serving process through a loopback HTTP
# request.  Site-wide proxy variables must not route that control RPC through
# an external proxy, otherwise multi-request object reclamation causes large
# host-side latency spikes after warmup.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export SGLANG_QWEN4_GFX90A_MQ4G128_ROUTED=${SGLANG_QWEN4_GFX90A_MQ4G128_ROUTED:-1}
export SGLANG_USE_AITER=${SGLANG_USE_AITER:-0}
export SGLANG_QWEN4_GFX90A_HC_DOWN_SPLIT=${SGLANG_QWEN4_GFX90A_HC_DOWN_SPLIT:-4}
export SGLANG_QWEN4_GFX90A_HC_SPLIT_REDUCE_IN_UP=${SGLANG_QWEN4_GFX90A_HC_SPLIT_REDUCE_IN_UP:-1}
export SGLANG_QWEN4_GFX90A_CK_HC_MIX_M32=${SGLANG_QWEN4_GFX90A_CK_HC_MIX_M32:-1}
# The correctness-verified expert-owned specializations cover M16--M128;
# reserve A4 grouped selection for larger uncaptured/eager tiers.
export SGLANG_QWEN4_GFX90A_MQ4G128_GROUPED_MIN_TOKENS=${SGLANG_QWEN4_GFX90A_MQ4G128_GROUPED_MIN_TOKENS:-256}
export SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M32=${SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M32:-1}
export SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M64=${SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M64:-1}
export SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M128=${SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M128:-1}
export SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M16=${SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M16:-1}
export SGLANG_QWEN4_GFX90A_MQ4G128_SYMMETRIC=${SGLANG_QWEN4_GFX90A_MQ4G128_SYMMETRIC:-1}

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --tp-size 4 \
  --ep-size 4 \
  --moe-a2a-backend none \
  --init-expert-location "${INIT_EXPERT_LOCATION}" \
  --ep-dispatch-algorithm dynamic \
  --attention-backend aiter \
  --linear-attn-backend triton \
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
  --chunked-prefill-size 1024 \
  --page-size 64 \
  --c128-page-size 16 \
  --swa-full-tokens-ratio 0.8 \
  --disable-radix-cache \
  --cuda-graph-bs-decode ${CUDA_GRAPH_BS_DECODE} \
  --host "${HOST}" \
  --port "${PORT}" \
  --skip-server-warmup \
  --watchdog-timeout 1200 \
  "$@"
