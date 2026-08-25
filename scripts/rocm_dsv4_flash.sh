#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SGLANG_DIR:-/home/pc/Code/sglang}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/pc/models/modelscope}"
DEFAULT_GPUS="4,5,6,7"
DEFAULT_PORT="30001"
DEFAULT_CUDA_GRAPH_MAX_BS_DECODE="8"
DEFAULT_CHUNKED_PREFILL_SIZE="2048"
DEFAULT_MORI_MAX_DISPATCH_TOKENS_PER_RANK="256"
DEFAULT_DISABLE_ATTN_TP_GATHER="1"
DEFAULT_MAX_TOTAL_TOKENS="8192"
DEFAULT_SWA_FULL_TOKENS_RATIO="0.65"
DEFAULT_MEM_FRACTION_STATIC="0.80"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-${DEFAULT_PORT}}"
BASE_URL="http://${HOST}:${PORT}"
COMMAND="${1:-}"
DSPARK_MODE=0
case "${COMMAND}" in
  start-dspark|serve-dspark|bench-dspark|bench-dspark-concurrent)
    DSPARK_MODE=1
    ;;
esac
if [[ "${DSPARK_MODE}" == "1" ]]; then
  LOG_FILE="${LOG_FILE:-/tmp/sglang_dsv4_flash_dspark.log}"
  PID_FILE="${PID_FILE:-/tmp/sglang_dsv4_flash_dspark.pid}"
else
  LOG_FILE="${LOG_FILE:-/tmp/sglang_dsv4_flash_ar.log}"
  PID_FILE="${PID_FILE:-/tmp/sglang_dsv4_flash_ar.pid}"
fi
PROFILE_DIR="${PROFILE_DIR:-/tmp/sglang_speed_profile_dsv4_ar}"
PROFILE_ID="${PROFILE_ID:-dsv4_ar_probe}"

export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/python:${ROOT_DIR}/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:${ROOT_DIR}/python/sglang/kernels/aot/python}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm/core-7.14}"
if [[ -d "${ROCM_ROOT}/include" ]]; then
  export CPATH="${ROCM_ROOT}/include${CPATH:+:${CPATH}}"
fi
if [[ -d "${ROCM_ROOT}/lib" ]]; then
  export LIBRARY_PATH="${ROCM_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${ROCM_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export DEEPEP_MODE="${DEEPEP_MODE:-normal}"
export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-0}"
export SGLANG_MORI_ASYNCLL_BLOCK_NUM="${SGLANG_MORI_ASYNCLL_BLOCK_NUM:-64}"
export SGLANG_MORI_ASYNCLL_WARP_NUM_PER_BLOCK="${SGLANG_MORI_ASYNCLL_WARP_NUM_PER_BLOCK:-8}"
export SGLANG_MORI_ASYNCLL_RDMA_BLOCK_NUM="${SGLANG_MORI_ASYNCLL_RDMA_BLOCK_NUM:-32}"
if [[ "${MORI_ENABLE_SDMA}" == "1" ]]; then
  MORI_APPLICATION_LIB="${MORI_APPLICATION_LIB:-/tmp/mori/python/mori/libmori_application.so}"
  if [[ ! -f "${MORI_APPLICATION_LIB}" ]]; then
    echo "MORI_ENABLE_SDMA=1 requires ${MORI_APPLICATION_LIB}" >&2
    exit 1
  fi
  # Mori statically embeds hsakmt. Preload it before PyTorch initializes HSA,
  # otherwise Anvil's second KFD state fails AcquireSystemProperties on ROCm.
  export LD_PRELOAD="${MORI_APPLICATION_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${DEFAULT_GPUS}}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_HACK_FLASHMLA_BACKEND="${SGLANG_HACK_FLASHMLA_BACKEND:-unified_kv_triton}"
export SGLANG_OPT_USE_AITER_MHC_PRE="${SGLANG_OPT_USE_AITER_MHC_PRE:-0}"
export SGLANG_OPT_USE_AITER_MHC_POST="${SGLANG_OPT_USE_AITER_MHC_POST:-0}"
export SGLANG_OPT_FUSE_MHC_POST_PRE="${SGLANG_OPT_FUSE_MHC_POST_PRE:-1}"
export SGLANG_OPT_USE_TRITON_MHC_COMBINE="${SGLANG_OPT_USE_TRITON_MHC_COMBINE:-1}"
export SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX="${SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX:-1}"
export SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A="${SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A:-1}"
export SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_MAX_BS="${SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_MAX_BS:-4}"
export SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A_SHADOW_REFERENCE="${SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A_SHADOW_REFERENCE:-0}"
export SGLANG_DSV4_GFX90A_MHC_PRE_MIX_COMPARE_REFERENCE="${SGLANG_DSV4_GFX90A_MHC_PRE_MIX_COMPARE_REFERENCE:-0}"
export SGLANG_OPT_USE_TRITON_INDEXER_POST="${SGLANG_OPT_USE_TRITON_INDEXER_POST:-1}"
export SGLANG_OPT_USE_TRITON_INDEXER_FULL="${SGLANG_OPT_USE_TRITON_INDEXER_FULL:-1}"
export SGLANG_DSV4_GFX90A_BF16_SHARED_EXPERT="${SGLANG_DSV4_GFX90A_BF16_SHARED_EXPERT:-0}"
export SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR="${SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR:-1}"
export SGLANG_DSV4_GFX90A_REPLICATE_EMBEDDING="${SGLANG_DSV4_GFX90A_REPLICATE_EMBEDDING:-0}"
export SGLANG_DSV4_GFX90A_FUSE_ATTN_INVERSE_ROPE="${SGLANG_DSV4_GFX90A_FUSE_ATTN_INVERSE_ROPE:-1}"
export SGLANG_DSV4_GFX90A_BF16_SHARED_GATE_UP="${SGLANG_DSV4_GFX90A_BF16_SHARED_GATE_UP:-1}"
export SGLANG_DSV4_GFX90A_BF16_SHARED_DOWN="${SGLANG_DSV4_GFX90A_BF16_SHARED_DOWN:-1}"
export SGLANG_DSV4_GFX90A_FUSED_SHARED_GATE_UP="${SGLANG_DSV4_GFX90A_FUSED_SHARED_GATE_UP:-1}"
export SGLANG_DSV4_GFX90A_WAVE64_SHARED_GATE_UP="${SGLANG_DSV4_GFX90A_WAVE64_SHARED_GATE_UP:-1}"
export SGLANG_DSV4_GFX90A_WAVE64_GEMV="${SGLANG_DSV4_GFX90A_WAVE64_GEMV:-1}"
export SGLANG_DSV4_GFX90A_WAVE64_FP32_GEMV="${SGLANG_DSV4_GFX90A_WAVE64_FP32_GEMV:-1}"
export SGLANG_DSV4_GFX90A_WAVE64_GROUPED_GEMV="${SGLANG_DSV4_GFX90A_WAVE64_GROUPED_GEMV:-1}"
export SGLANG_DSV4_GFX90A_INT8_WEIGHT_GEMV="${SGLANG_DSV4_GFX90A_INT8_WEIGHT_GEMV:-0}"
export SGLANG_DSV4_GFX90A_WAVE64_MHC_PRE_MIX="${SGLANG_DSV4_GFX90A_WAVE64_MHC_PRE_MIX:-1}"
export SGLANG_DSV4_GFX90A_NATIVE_MHC_SINKHORN="${SGLANG_DSV4_GFX90A_NATIVE_MHC_SINKHORN:-1}"
export SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS="${SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS:-8}"
export SGLANG_DSV4_GFX90A_FUSED_MHC_WEIGHTED_RMS="${SGLANG_DSV4_GFX90A_FUSED_MHC_WEIGHTED_RMS:-1}"
export SGLANG_DSV4_GFX90A_SPLITK_MHC_PRE_MIX="${SGLANG_DSV4_GFX90A_SPLITK_MHC_PRE_MIX:-1}"
export SGLANG_DSV4_GFX90A_FP16_MHC_DOT="${SGLANG_DSV4_GFX90A_FP16_MHC_DOT:-1}"
export SGLANG_DSV4_GFX90A_FUSED_MHC_SPLITK_TAIL="${SGLANG_DSV4_GFX90A_FUSED_MHC_SPLITK_TAIL:-1}"
export SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE="${SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE:-0}"
export SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE_FULL="${SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE_FULL:-0}"
export SGLANG_DSV4_GFX90A_MHC_FAST_MATH="${SGLANG_DSV4_GFX90A_MHC_FAST_MATH:-0}"
export SGLANG_DSV4_GFX90A_FUSE_MHC_POST_RMS_PARTIALS="${SGLANG_DSV4_GFX90A_FUSE_MHC_POST_RMS_PARTIALS:-1}"
export SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER="${SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER:-0}"
export SGLANG_DSV4_GFX90A_TRITON_TOPK_ROUTER="${SGLANG_DSV4_GFX90A_TRITON_TOPK_ROUTER:-0}"
export SGLANG_DSV4_GFX90A_ROUTER_NUM_WARPS="${SGLANG_DSV4_GFX90A_ROUTER_NUM_WARPS:-1}"
export SGLANG_MORI_NO_PAD_MASK="${SGLANG_MORI_NO_PAD_MASK:-1}"
export SGLANG_DSV4_GFX90A_MORI_SHARED_EXPERT_TP="${SGLANG_DSV4_GFX90A_MORI_SHARED_EXPERT_TP:-1}"
export SGLANG_GFX90A_AITER_MORI_SKIP_PREQUANT="${SGLANG_GFX90A_AITER_MORI_SKIP_PREQUANT:-1}"
export SGLANG_USE_AITER_MOE_GU_ITLV="${SGLANG_USE_AITER_MOE_GU_ITLV:-1}"
if [[ "${SGLANG_DSV4_GFX90A_MORI_SHARED_EXPERT_TP}" == "1" ]]; then
  export SGLANG_DSV4_MORI_ROTATE_SHARED_EXPERT_OWNER="${SGLANG_DSV4_MORI_ROTATE_SHARED_EXPERT_OWNER:-0}"
else
  export SGLANG_DSV4_MORI_ROTATE_SHARED_EXPERT_OWNER="${SGLANG_DSV4_MORI_ROTATE_SHARED_EXPERT_OWNER:-1}"
fi
export MORI_DISABLE_TOPO="${MORI_DISABLE_TOPO:-1}"
export MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-1}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-6G}"
export SGLANG_MORI_DISPATCH_DTYPE="${SGLANG_MORI_DISPATCH_DTYPE:-bf16}"
export SGLANG_MORI_COMBINE_DTYPE="${SGLANG_MORI_COMBINE_DTYPE:-bf16}"
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-${DEFAULT_MORI_MAX_DISPATCH_TOKENS_PER_RANK}}"
export SGLANG_MORI_USE_EXTERNAL_INP_BUF="${SGLANG_MORI_USE_EXTERNAL_INP_BUF:-0}"
export SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK:-16}"
export SGLANG_MORI_DECODE_TIERED_CAPACITY="${SGLANG_MORI_DECODE_TIERED_CAPACITY:-0}"
export SGLANG_MORI_ASYNCLL_DECODE_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_ASYNCLL_DECODE_MAX_DISPATCH_TOKENS_PER_RANK:-0}"
export SGLANG_MORI_MOE_MAX_INPUT_TOKENS="${SGLANG_MORI_MOE_MAX_INPUT_TOKENS:-0}"
export SGLANG_ROCM_CUDA_GRAPH_UPLOAD="${SGLANG_ROCM_CUDA_GRAPH_UPLOAD:-0}"
export SGLANG_MORI_INTRANODE_BLOCK_NUM="${SGLANG_MORI_INTRANODE_BLOCK_NUM:-32}"
export SGLANG_MORI_INTRANODE_WARP_NUM_PER_BLOCK="${SGLANG_MORI_INTRANODE_WARP_NUM_PER_BLOCK:-8}"
export SGLANG_MORI_INTRANODE_COMBINE_BLOCK_NUM="${SGLANG_MORI_INTRANODE_COMBINE_BLOCK_NUM:-32}"
export SGLANG_MORI_INTRANODE_COMBINE_WARP_NUM_PER_BLOCK="${SGLANG_MORI_INTRANODE_COMBINE_WARP_NUM_PER_BLOCK:-4}"
export AITER_GFX90A_MXFP4_QUANT_MAX_ROWS="${AITER_GFX90A_MXFP4_QUANT_MAX_ROWS:-64}"
export SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT="${SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT:-0}"
export SGLANG_DSV4_GFX90A_AITER_MOE_STAGE2_64THREAD="${SGLANG_DSV4_GFX90A_AITER_MOE_STAGE2_64THREAD:-0}"
# TP8's 8 KiB peer-read reductions naturally select eight CTAs, but the
# gfx90a/AIter signal protocol is only correctness-stable at four CTAs. TP4
# already selects four without an override. Keep an explicit user value for
# geometry experiments and scope the safe default to TP8/EP1 no-A2A.
if [[ -z "${AITER_GFX90A_AR_SMALL_BLOCKS+x}" && \
      "${TP_SIZE:-4}" == "8" && "${EP_SIZE:-4}" == "1" && \
      "${MOE_A2A_BACKEND:-mori}" == "none" ]]; then
  export AITER_GFX90A_AR_SMALL_BLOCKS=4
fi
# The direct FP4 decode kernel is beneficial only for TP4/EP1's K=512 down
# shard. EP2/EP4 keep CKTile: their wider K already fills wave64 and the
# subgroup protocol adds overhead.
if [[ "${EP_SIZE:-4}" == "1" ]]; then
  DEFAULT_GFX90A_FP4_DIRECT_MOE=1
else
  DEFAULT_GFX90A_FP4_DIRECT_MOE=0
fi
if [[ "${EP_SIZE:-4}" == "1" && "${MOE_A2A_BACKEND:-mori}" == "none" ]]; then
  DEFAULT_GFX90A_MHC_TP_ONLY_GEOMETRY=1
else
  DEFAULT_GFX90A_MHC_TP_ONLY_GEOMETRY=0
fi
export SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE="${SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE:-${DEFAULT_GFX90A_FP4_DIRECT_MOE}}"
export SGLANG_DSV4_GFX90A_FP4_GROUPED_PREFILL="${SGLANG_DSV4_GFX90A_FP4_GROUPED_PREFILL:-${DEFAULT_GFX90A_FP4_DIRECT_MOE}}"
if [[ "${TP_SIZE:-4}" == "4" && "${EP_SIZE:-4}" == "1" && \
      "${MOE_A2A_BACKEND:-mori}" == "none" ]]; then
  DEFAULT_GFX90A_FP4_MFMA32_PREFILL=1
else
  DEFAULT_GFX90A_FP4_MFMA32_PREFILL=0
fi
export SGLANG_DSV4_GFX90A_FP4_MFMA32_PREFILL="${SGLANG_DSV4_GFX90A_FP4_MFMA32_PREFILL:-${DEFAULT_GFX90A_FP4_MFMA32_PREFILL}}"
export SGLANG_DSV4_GFX90A_FP4_MFMA64_PREFILL="${SGLANG_DSV4_GFX90A_FP4_MFMA64_PREFILL:-${DEFAULT_GFX90A_FP4_MFMA32_PREFILL}}"
export SGLANG_DSV4_GFX90A_MHC_TP_ONLY_GEOMETRY="${SGLANG_DSV4_GFX90A_MHC_TP_ONLY_GEOMETRY:-${DEFAULT_GFX90A_MHC_TP_ONLY_GEOMETRY}}"

# AIter may optionally shrink the fixed Mori MXFP4 quantization grid.  DSV4
# routes six experts per token, so the static grid must cover every row that a
# captured decode tier can make live.  Zero keeps AIter's fully general grid.
if [[ "${AITER_GFX90A_MXFP4_QUANT_MAX_ROWS:-0}" -gt 0 ]]; then
  required_quant_rows=$(( ${CUDA_GRAPH_MAX_BS_DECODE:-${DEFAULT_CUDA_GRAPH_MAX_BS_DECODE}} * 6 ))
  if [[ "${AITER_GFX90A_MXFP4_QUANT_MAX_ROWS}" -lt "${required_quant_rows}" ]]; then
    echo "error: AITER_GFX90A_MXFP4_QUANT_MAX_ROWS=${AITER_GFX90A_MXFP4_QUANT_MAX_ROWS} is below graph_bs*topk=${required_quant_rows}" >&2
    exit 2
  fi
fi
required_dispatch_rows=$(( ${CUDA_GRAPH_MAX_BS_DECODE:-${DEFAULT_CUDA_GRAPH_MAX_BS_DECODE}} * 6 ))
if [[ "${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}" -lt "${required_dispatch_rows}" ]]; then
  echo "error: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK} is below graph_bs*topk=${required_dispatch_rows}" >&2
  exit 2
fi
if [[ "${SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK:-0}" -gt 0 ]] && \
   [[ "${SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK}" -lt "$(( ${CUDA_GRAPH_MAX_BS_DECODE:-${DEFAULT_CUDA_GRAPH_MAX_BS_DECODE}} * 2 ))" ]]; then
  echo "error: SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK} is below the validated 2*graph_bs floor=$(( ${CUDA_GRAPH_MAX_BS_DECODE:-${DEFAULT_CUDA_GRAPH_MAX_BS_DECODE}} * 2 ))" >&2
  exit 2
fi

server_args=(
  --model-path "${MODEL_PATH}"
  --load-format safetensors
  --tp-size "${TP_SIZE:-4}"
  --ep-size "${EP_SIZE:-4}"
  --moe-a2a-backend "${MOE_A2A_BACKEND:-mori}"
  --moe-runner-backend "${MOE_RUNNER_BACKEND:-aiter}"
  --attention-backend "${ATTENTION_BACKEND:-dsv4}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE:-${DEFAULT_CUDA_GRAPH_MAX_BS_DECODE}}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-${DEFAULT_CHUNKED_PREFILL_SIZE}}"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-${DEFAULT_MAX_TOTAL_TOKENS}}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-${DEFAULT_SWA_FULL_TOKENS_RATIO}}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-${DEFAULT_MEM_FRACTION_STATIC}}"
  --skip-server-warmup
  --enable-tokenizer-batch-encode
  # Per-token scheduler logging measurably stalls CPU graph replay on the
  # single-request latency path. Keep periodic observability without putting
  # formatted I/O on every generated token.
  --decode-log-interval "${DECODE_LOG_INTERVAL:-10000}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
)
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  server_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${CPU_OFFLOAD_GB:-}" ]]; then
  server_args+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  server_args+=(--served-model-name "${SERVED_MODEL_NAME}")
fi
if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
  server_args+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi
if [[ -n "${REASONING_PARSER:-}" ]]; then
  server_args+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${CUDA_GRAPH_BS_DECODE:-}" ]]; then
  read -r -a cuda_graph_bs_decode <<<"${CUDA_GRAPH_BS_DECODE}"
  server_args+=(--cuda-graph-bs-decode "${cuda_graph_bs_decode[@]}")
fi
if [[ "${PRE_WARM_NCCL:-${DSPARK_MODE}}" == "1" ]]; then
  server_args+=(--pre-warm-nccl)
fi
# Scheduler overlap plus the single-batch fast path keeps graph replay fed on
# the latency-critical native-AR path. Set this to 1 only for the legacy A/B.
if [[ "${DISABLE_OVERLAP_SCHEDULE:-0}" == "1" ]]; then
  server_args+=(--disable-overlap-schedule)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  server_args+=(--disable-custom-all-reduce)
fi
if [[ "${DISABLE_ATTN_TP_GATHER:-${DEFAULT_DISABLE_ATTN_TP_GATHER}}" == "1" ]]; then
  server_args+=(--disable-attn-tp-gather)
fi
if [[ -n "${DEEPEP_MODE:-}" ]]; then
  server_args+=(--deepep-mode "${DEEPEP_MODE}")
fi
if [[ "${EP_SIZE:-4}" == "1" ]]; then
  DEFAULT_ENABLE_SINGLE_BATCH_OVERLAP=0
else
  DEFAULT_ENABLE_SINGLE_BATCH_OVERLAP=1
fi
if [[ "${ENABLE_SINGLE_BATCH_OVERLAP:-${DEFAULT_ENABLE_SINGLE_BATCH_OVERLAP}}" == "1" ]]; then
  server_args+=(--enable-single-batch-overlap)
fi
if [[ "${ENABLE_PROFILE_CUDA_GRAPH:-0}" == "1" ]]; then
  export SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE="${SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE:-1}"
  server_args+=(--enable-profile-cuda-graph)
fi
if [[ "${DISABLE_DECODE_CUDA_GRAPH:-0}" == "1" ]]; then
  server_args+=(--disable-decode-cuda-graph)
fi
speculative_env_vars=(
  SPECULATIVE_ALGORITHM
  SPECULATIVE_DRAFT_MODEL_PATH
  SPECULATIVE_DSPARK_BLOCK_SIZE
  SPECULATIVE_DSPARK_SPS_TABLE_PATH
  SPECULATIVE_DSPARK_CONFIDENCE_STS_PATH
  SPECULATIVE_DSPARK_ALIGN_VERIFY_TOKENS_TO_GRAPH_TIER
)
if [[ "${DSPARK_MODE}" == "1" ]]; then
  export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
  # The fixed short benchmark never crosses index_topk=512. Avoid retaining a
  # second full-indexer graph for every 6-token verify tier; BS16 otherwise
  # exceeds MI250X memory before the service starts.
  export SGLANG_DSV4_DSA_DENSE_ONLY_GRAPH="${SGLANG_DSV4_DSA_DENSE_ONLY_GRAPH:-1}"
  server_args+=(
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "${SPECULATIVE_DSPARK_BLOCK_SIZE:-5}"
  )
else
  for var_name in "${speculative_env_vars[@]}"; do
    if [[ -n "${!var_name:-}" && "${!var_name}" != "0" ]]; then
      echo "error: ${var_name} is set; this harness only permits native AR decode" >&2
      exit 2
    fi
  done
fi

usage() {
  cat <<'EOF'
Usage: scripts/rocm_dsv4_flash.sh <command> [args]

This script measures the single-request AR path only. It defaults to GPU 4-7
and port 30001, and never sends a batched payload. CUDA graph capture may keep
larger prepared tiers; that does not change the request batch size.

Commands:
  start                 Start the configured ROCm DSV4 service in background.
  serve                 Run the configured ROCm DSV4 service in foreground.
  stop                  Stop the background service from the pid file.
  status                Show service process and ROCm VRAM/PID state.
  logs [n]              Tail the last n log lines, default 120.
  bench [tokens] [reps] Run official-prompt single-request AR probe.
                        Defaults: tokens=256, reps=1.
  bench-context [words] [tokens] [reps]
                        Run native AR beyond DSV4's dense indexer threshold.
                        Defaults: words=2300, tokens=128, reps=3.
  bench-concurrent [tokens] [requests] [reps]
                        Run independent native-AR requests concurrently.
                        Defaults: tokens=256, requests=4, reps=1.
  start-dspark          Start a separately labelled DSpark service.
  serve-dspark          Run the separately labelled DSpark service in foreground.
  bench-dspark [tokens] [reps]
                        Run one DSpark request and report emitted-token throughput.
  bench-dspark-concurrent [tokens] [requests] [reps]
                        Run concurrent DSpark requests and report emitted-token
                        throughput plus per-response acceptance statistics.
  profile [tokens] [steps]
                        Start SGLang stage profiler, then send one request.
                        Defaults: tokens=32, steps=1.
                        ROCprofiler may segfault after stop on this stack.
  parse-profile [dir]   Summarize kernels and steady per-layer decode time.

AR-only contract:
  Any SPECULATIVE_* setting is rejected before the command runs. This harness
  always measures one model forward per generated token.

DSpark contract:
  Only the explicitly named *-dspark commands enable speculative decoding.
  Their throughput is based on tokens actually returned to clients; it is not
  comparable to the native-AR forward-per-token contract above.

Optional env:
  TP_SIZE=4 EP_SIZE=4 MOE_A2A_BACKEND=mori
                              # Default EP4 service. For the validated BS1
                              # no-A2A oracle use EP_SIZE=1 and
                              # MOE_A2A_BACKEND=none. That profile requires the
                              # AIter system-barrier patch under
                              # scripts/rocm/patches/.
  SGLANG_MORI_ALLOW_PARTIAL_EP=0
                              # Honor EP_SIZE smaller than TP_SIZE for the
                              # experimental hybrid expert EP/TP layout. The
                              # validated TP4/EP2 profile uses Mori dispatch
                              # and combine blocks=16 plus the Mori subgroup
                              # bootstrap patch under scripts/rocm/patches/.
  DISABLE_ATTN_TP_GATHER=1   # single lane default; set 0 to restore padded graph capture.
  SGLANG_MORI_INTRANODE_BLOCK_NUM=<n>
  SGLANG_MORI_INTRANODE_WARP_NUM_PER_BLOCK=<n>
                              # Override Mori intra-node launch geometry for probes.
  SGLANG_MORI_INTRANODE_COMBINE_BLOCK_NUM=32
  SGLANG_MORI_INTRANODE_COMBINE_WARP_NUM_PER_BLOCK=4
                              # Independently tune zero-copy combine. The defaults
                              # are validated for EP4, one-token decode on gfx90a.
  SGLANG_MORI_USE_EXTERNAL_INP_BUF=0
                              # Direct AIter stage-2 output into Mori's registered
                              # peer-read buffer. Enabled by default; set 1 for push.
  SGLANG_USE_AITER_MOE_GU_ITLV=1
                              # AIter clamped-SwiGLU interleave. Set 0 only with
                              # experimental FP4 Mori dispatch passthrough.
  AITER_GFX90A_MXFP4_QUANT_MAX_ROWS=0
                              # Optional fixed Mori quant-grid bound. Must be at
                              # least CUDA_GRAPH_MAX_BS_DECODE*6; zero is general.
  SGLANG_DSV4_GFX90A_BF16_SHARED_EXPERT=1
                              # Opt in to the experimental shared-expert BF16
                              # weight cache. It is not graph-capture stable yet.
  SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A=1
                              # Use the gfx90a Triton MHC pre-mix with Mori/A2A.
                              # Graph warmups consume the reference result;
                              # the captured/replayed graph uses the fast kernel.
  SGLANG_DSV4_GFX90A_MHC_PRE_MIX_COMPARE_REFERENCE=1
                              # One-shot eager diagnostic: log custom-vs-FP32
                              # MHC pre-mix error. Default off.
  SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR=1
                              # Cache the three decode attention projections as
                              # BF16 on gfx90a. Enabled by default; costs ~1 GiB/GPU.
  SGLANG_DSV4_GFX90A_REPLICATE_EMBEDDING=0
                              # Replicate input embeddings and remove the first
                              # TP all-reduce. Experimental; costs ~0.75 GiB/GPU.
  SGLANG_DSV4_GFX90A_BF16_SHARED_GATE_UP=1
  SGLANG_DSV4_GFX90A_BF16_SHARED_DOWN=1
                              # Cache both shared-expert projections as BF16 on
                              # gfx90a. Both are enabled by default with the 4096
                              # token / 0.80 static-memory decode profile.
  SGLANG_DSV4_GFX90A_FUSED_SHARED_GATE_UP=1
                              # Fuse the single-token BF16 gate/up projection
                              # with DSV4's bounded SwiGLU on the owner rank.
  SGLANG_DSV4_GFX90A_MORI_SHARED_EXPERT_TP=1
                              # Partition Mori's DSV4 shared expert across TP,
                              # overlap it with routed MoE, then use one peer AR
                              # for shared sum + routed gather. Default enabled.
                              # Keep DISABLE_ATTN_TP_GATHER=1 for graph BS=1.
  SGLANG_DSV4_MORI_ROTATE_SHARED_EXPERT_OWNER=1
                              # Rotate each layer's real token chunk across the
                              # four Mori ranks. Defaults off with shared TP (the
                              # TP shards are already balanced), on with the TP1
                              # shared-expert fallback.
  DISABLE_CUSTOM_ALL_REDUCE=0 # Use the fixed AIter peer-read custom AR.
  SERVED_MODEL_NAME=<name>     # Stable OpenAI-compatible model id returned by /v1/models.
  TOOL_CALL_PARSER=deepseekv4 # Parse DSV4 DSML output into OpenAI tool_calls.
  REASONING_PARSER=deepseek-v4
                              # Split DSV4 reasoning_content for agent clients.
  DEEPEP_MODE=low_latency     # Experimental Mori AsyncLL split-phase transport.
  MORI_ENABLE_SDMA=1          # Move AsyncLL transport to copy engines.
  SGLANG_MORI_ASYNCLL_BLOCK_NUM=64
  SGLANG_MORI_ASYNCLL_WARP_NUM_PER_BLOCK=8
  SGLANG_MORI_ASYNCLL_RDMA_BLOCK_NUM=32
                              # AsyncLL geometry; gfx90a BS1 experiments may
                              # override block count without changing normal Mori.
  MORI_APPLICATION_LIB=/tmp/mori/python/mori/libmori_application.so
                              # Preloaded before PyTorch so Anvil and HSA share
                              # one KFD thunk state. For the old path set both
                              # DEEPEP_MODE=normal and MORI_ENABLE_SDMA=0.
  ENABLE_SINGLE_BATCH_OVERLAP=1
                              # Overlap communication inside one request; this
                              # does not create a multi-request batch.
  ENABLE_PROFILE_CUDA_GRAPH=1 # Record kernels while the decode graph is captured.
  SGLANG_ROCM_CUDA_GRAPH_UPLOAD=0
                              # Explicitly hipGraphUpload each captured graph.
                              # Experimental; ROCm ignores instantiate upload flags.
  DISABLE_DECODE_CUDA_GRAPH=1 # Eager-only operator probe; not an AR benchmark mode.
EOF
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

wait_ready() {
  local timeout="${WAIT_READY_TIMEOUT:-900}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if grep -q "The server is fired up and ready to roll" "${LOG_FILE}" 2>/dev/null && "${PYTHON_BIN}" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=2):
    pass
PY
    then
      # The HTTP process can accept TCP before the tokenizer manager has
      # installed its scheduler RPC state.  Freeze only after the server's own
      # ready marker, and retry here rather than during application startup.
      local freeze_ok=0
      for _ in {1..10}; do
        if curl --noproxy '*' -sS --fail --max-time 15 -X POST \
          "${BASE_URL}/freeze_gc" >/dev/null 2>&1; then
          freeze_ok=1
          break
        fi
        sleep 1
      done
      if [[ "${freeze_ok}" == "1" ]]; then
        echo "ready: ${BASE_URL} (GC frozen)"
      else
        echo "ready: ${BASE_URL} (warning: freeze_gc did not succeed)" >&2
      fi
      return 0
    fi
    if ! is_running; then
      echo "server exited during startup; tail follows:" >&2
      tail -120 "${LOG_FILE}" >&2 || true
      return 1
    fi
    if (( "$(date +%s)" - start_ts > timeout )); then
      echo "timed out waiting for server readiness after ${timeout}s" >&2
      tail -120 "${LOG_FILE}" >&2 || true
      return 1
    fi
    sleep 5
  done
}

start_server() {
  if is_running; then
    echo "already running pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
    return 0
  fi
  mkdir -p "$(dirname "${LOG_FILE}")"
  : > "${LOG_FILE}"
  cd "${ROOT_DIR}"
  nohup "${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}" \
    >>"${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "started pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
  wait_ready
}

serve_server() {
  cd "${ROOT_DIR}"
  exec "${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}"
}

stop_server() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "stopped pid=${pid}"
  else
    rm -f "${PID_FILE}"
    echo "not running"
  fi
}

status() {
  if is_running; then
    ps -p "$(cat "${PID_FILE}")" -o pid,stat,cmd
  else
    echo "pid-file service not running"
  fi
  ps -eo pid,stat,cmd | rg 'sglang.launch_server|sglang::schedul|scheduler_TP' || true
  rocm-smi --showpids --showmeminfo vram || true
}

bench() {
  local tokens="${1:-256}"
  local reps="${2:-1}"
  "${PYTHON_BIN}" - "${BASE_URL}" "${tokens}" "${reps}" <<'PY'
import hashlib
import json
import sys
import time
import urllib.request

base_url, tokens, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
url = base_url + "/generate"
prompt = "<｜begin▁of▁sentence｜><｜User｜>Explain why 2+2=4 in one sentence.<｜Assistant｜><think>"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def send(payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with opener.open(req, timeout=timeout) as response:
        body = response.read()
    return time.perf_counter() - start, json.loads(body)

for rep in range(reps):
    salt_ns = time.time_ns()
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": tokens,
        "ignore_eos": True,
    }
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "cache_salt": f"ar-{tokens}-{rep}-{salt_ns}",
    }
    print(f"BEGIN rep={rep} tokens={tokens} mode=single-request-AR", flush=True)
    dt, out = send(payload, timeout=max(1200, tokens * 2))
    output_tokens = len(out.get("output_ids") or [])
    output_hash = hashlib.sha256(
        json.dumps(out.get("output_ids") or [], separators=(",", ":")).encode()
    ).hexdigest()[:16]
    first_text = (out.get("text") or "")[:96]
    reason = out.get("meta_info", {}).get("finish_reason")
    print(
        "END "
        f"rep={rep} wall={dt:.3f}s output_tokens={output_tokens} "
        f"ar_tok/s={output_tokens / dt:.3f} "
        f"ar_ms/token={(dt / max(output_tokens, 1)) * 1000:.1f} "
        f"finish={reason} "
        f"output_sha256={output_hash} "
        f"text0={first_text!r}",
        flush=True,
    )
PY
}

bench_context() {
  local words="${1:-2300}"
  local tokens="${2:-128}"
  local reps="${3:-3}"
  "${PYTHON_BIN}" - "${BASE_URL}" "${words}" "${tokens}" "${reps}" <<'PY'
import hashlib
import json
import sys
import time
import urllib.request

base_url = sys.argv[1]
words, tokens, reps = map(int, sys.argv[2:])
url = base_url + "/generate"
# Repetition is intentional: cache_salt prevents prefix-cache reuse, while the
# server-reported prompt token count makes every long-context result auditable.
prompt = "<｜begin▁of▁sentence｜><｜User｜>" + " indexer" * words + "<｜Assistant｜><think>"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

for rep in range(reps):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": tokens,
            "ignore_eos": True,
        },
        "cache_salt": f"ar-context-{words}-{tokens}-{rep}-{time.time_ns()}",
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_chunk_at = None
    last_chunk_at = None
    out = {}
    with opener.open(req, timeout=max(1200, tokens * 4)) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[len("data:") :].strip()
            if payload_text == "[DONE]":
                break
            out = json.loads(payload_text)
            now = time.perf_counter()
            first_chunk_at = now if first_chunk_at is None else first_chunk_at
            last_chunk_at = now
    end = time.perf_counter()
    dt = end - start
    output_tokens = len(out.get("output_ids") or [])
    output_hash = hashlib.sha256(
        json.dumps(out.get("output_ids") or [], separators=(",", ":")).encode()
    ).hexdigest()[:16]
    meta = out.get("meta_info", {})
    prompt_tokens = meta.get(
        "prompt_tokens", meta.get("input_token_logprobs", "unknown")
    )
    decode_dt = max((last_chunk_at or end) - (first_chunk_at or end), 1e-9)
    decode_tokens = max(output_tokens - 1, 0)
    print(
        f"END rep={rep} prompt_tokens={prompt_tokens} output_tokens={output_tokens} "
        f"wall={dt:.3f}s ttft={(first_chunk_at - start):.3f}s "
        f"decode_tok/s={decode_tokens / decode_dt:.3f} "
        f"finish={meta.get('finish_reason')} output_sha256={output_hash}",
        flush=True,
    )
PY
}

bench_concurrent() {
  local tokens="${1:-256}"
  local requests="${2:-4}"
  local reps="${3:-1}"
  local expected_hash="${EXPECTED_OUTPUT_SHA256:-f3060e252a69f624}"
  "${PYTHON_BIN}" - "${BASE_URL}" "${tokens}" "${requests}" "${reps}" "${expected_hash}" <<'PY'
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
import urllib.request

base_url = sys.argv[1]
tokens, requests, reps = map(int, sys.argv[2:5])
expected_hash = sys.argv[5]
url = base_url + "/generate"
prompt = "<｜begin▁of▁sentence｜><｜User｜>Explain why 2+2=4 in one sentence.<｜Assistant｜><think>"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def send(payload, start_barrier):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start_barrier.wait()
    begin = time.perf_counter()
    with opener.open(req, timeout=max(1200, tokens * 2)) as response:
        body = response.read()
    return time.perf_counter() - begin, json.loads(body)

for rep in range(reps):
    barrier = threading.Barrier(requests + 1)
    salt_ns = time.time_ns()
    payloads = [
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
            "cache_salt": f"ar-concurrent-{tokens}-{requests}-{rep}-{index}-{salt_ns}",
        }
        for index in range(requests)
    ]
    print(
        f"BEGIN rep={rep} tokens={tokens} requests={requests} "
        "mode=concurrent-independent-AR",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
        futures = [pool.submit(send, payload, barrier) for payload in payloads]
        barrier.wait()
        group_begin = time.perf_counter()
        results = [future.result() for future in futures]
        group_wall = time.perf_counter() - group_begin

    output_tokens = [len(out.get("output_ids") or []) for _, out in results]
    output_hashes = [
        hashlib.sha256(
            json.dumps(out.get("output_ids") or [], separators=(",", ":")).encode()
        ).hexdigest()[:16]
        for _, out in results
    ]
    reasons = [out.get("meta_info", {}).get("finish_reason") for _, out in results]
    request_walls = [wall for wall, _ in results]
    total_tokens = sum(output_tokens)
    print(
        "END "
        f"rep={rep} group_wall={group_wall:.3f}s total_tokens={total_tokens} "
        f"aggregate_ar_tok/s={total_tokens / group_wall:.3f} "
        f"per_request_ar_tok/s={[round(n / wall, 3) for n, wall in zip(output_tokens, request_walls)]} "
        f"output_tokens={output_tokens} output_sha256={output_hashes} "
        f"hashes_match={len(set(output_hashes)) == 1} "
        f"reference_match={all(h == expected_hash for h in output_hashes)} "
        f"finish={reasons}",
        flush=True,
    )
PY
}

bench_dspark_concurrent() {
  local tokens="${1:-256}"
  local requests="${2:-4}"
  local reps="${3:-1}"
  "${PYTHON_BIN}" - "${BASE_URL}" "${tokens}" "${requests}" "${reps}" <<'PY'
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
import urllib.request

base_url = sys.argv[1]
tokens, requests, reps = map(int, sys.argv[2:5])
generate_url = base_url + "/generate"
server_info_url = base_url + "/server_info"
prompt = "<｜begin▁of▁sentence｜><｜User｜>Explain why 2+2=4 in one sentence.<｜Assistant｜><think>"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def server_accept_length():
    try:
        with opener.open(server_info_url, timeout=30) as response:
            info = json.loads(response.read())
        states = info.get("internal_states") or []
        values = [
            state.get("avg_spec_accept_length")
            for state in states
            if state.get("avg_spec_accept_length") is not None
        ]
        return values
    except Exception as exc:
        return [f"unavailable:{exc}"]

def send(payload, start_barrier):
    req = urllib.request.Request(
        generate_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_barrier.wait()
    begin = time.perf_counter()
    with opener.open(req, timeout=max(1200, tokens * 2)) as response:
        body = response.read()
    return time.perf_counter() - begin, json.loads(body)

for rep in range(reps):
    barrier = threading.Barrier(requests + 1)
    salt_ns = time.time_ns()
    payloads = [
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
            "cache_salt": f"dspark-{tokens}-{requests}-{rep}-{index}-{salt_ns}",
        }
        for index in range(requests)
    ]
    accept_before = server_accept_length()
    print(
        f"BEGIN rep={rep} tokens={tokens} requests={requests} "
        f"mode=DSpark-gamma5 accept_before={accept_before}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
        futures = [pool.submit(send, payload, barrier) for payload in payloads]
        barrier.wait()
        group_begin = time.perf_counter()
        results = [future.result() for future in futures]
        group_wall = time.perf_counter() - group_begin

    output_tokens = [len(out.get("output_ids") or []) for _, out in results]
    output_hashes = [
        hashlib.sha256(
            json.dumps(out.get("output_ids") or [], separators=(",", ":")).encode()
        ).hexdigest()[:16]
        for _, out in results
    ]
    request_walls = [wall for wall, _ in results]
    reasons = [out.get("meta_info", {}).get("finish_reason") for _, out in results]
    spec_stats = [
        {
            key: out.get("meta_info", {}).get(key)
            for key in (
                "spec_accept_length",
                "spec_accept_rate",
                "spec_accepted_drafts",
                "spec_verify_ct",
            )
            if out.get("meta_info", {}).get(key) is not None
        }
        for _, out in results
    ]
    total_tokens = sum(output_tokens)
    accept_after = server_accept_length()
    print(
        "END "
        f"rep={rep} group_wall={group_wall:.3f}s total_emitted_tokens={total_tokens} "
        f"aggregate_dspark_tok/s={total_tokens / group_wall:.3f} "
        f"per_request_dspark_tok/s="
        f"{[round(n / wall, 3) for n, wall in zip(output_tokens, request_walls)]} "
        f"output_tokens={output_tokens} output_sha256={output_hashes} "
        f"hashes_match={len(set(output_hashes)) == 1} "
        f"finish={reasons} spec_stats={spec_stats} "
        f"avg_spec_accept_length={accept_after}",
        flush=True,
    )
PY
}

profile() {
  local tokens="${1:-32}"
  local steps="${2:-1}"
  rm -rf "${PROFILE_DIR}"
  mkdir -p "${PROFILE_DIR}"
  curl --noproxy '*' -sS --max-time 30 -X POST "${BASE_URL}/start_profile" \
    -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"${PROFILE_DIR}\",\"profile_id\":\"${PROFILE_ID}\",\"num_steps\":${steps},\"profile_by_stage\":true,\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false}"
  echo
  bench "${tokens}" 1
}

parse_profile() {
  local trace_dir="${1:-${PROFILE_DIR}}"
  "${PYTHON_BIN}" - "${trace_dir}" <<'PY'
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys

trace_dir = sys.argv[1]

def category(name):
    if "_gemm_a16_w16_gated_kernel" in name:
        return "shared bounded gate/up GEMM"
    if (
        "cross_device_reduce" in name
        or "reduce_scatter_first_dim" in name
        or "allgather_vec" in name
        or "allgather_naive" in name
    ):
        return "AIter peer collectives"
    if "_hc_split_sinkhorn4" in name:
        return "MHC split sinkhorn Triton"
    if "_mhc_weighted_sum" in name or "_mhc_post_combine" in name:
        return "MHC pre/post Triton combine"
    if "_dynamic_mxfp4_quant" in name:
        return "MXFP4 activation quant"
    if "_w8a8_block_fp8_matmul" in name:
        return "dense W8A8 FP8 matmul"
    if "EpDispatchIntraNodeKernel" in name:
        return "Mori dispatch"
    if "EpCombineIntraNodeKernel" in name:
        return "Mori combine"
    if "EpDispatchLowLatencyAsyncSendCopy" in name:
        return "Mori AsyncLL dispatch send copy"
    if "EpDispatchLowLatencyAsyncSendTransfer" in name:
        return "Mori AsyncLL dispatch SDMA"
    if "EpDispatchLowLatencyAsyncRecvTransfer" in name:
        return "Mori AsyncLL dispatch recv wait"
    if "EpDispatchLowLatencyAsyncRecvCopy" in name:
        return "Mori AsyncLL dispatch recv copy"
    if "EpCombineLowLatencyAsyncSendCopy" in name:
        return "Mori AsyncLL combine send copy"
    if "EpCombineLowLatencyAsyncSendTransfer" in name:
        return "Mori AsyncLL combine SDMA"
    if "EpCombineLowLatencyAsyncRecvTransfer" in name:
        return "Mori AsyncLL combine recv wait"
    if "EpCombineLowLatencyAsyncRecvCopy" in name:
        return "Mori AsyncLL combine recv copy"
    if "ck::kernel_moe_mxgemm" in name or "moe_ck" in name:
        return "AIter CK FP4 MoE GEMM"
    if "_moe_mxfp4_sort" in name:
        return "MoE mxfp4 sort"
    if "_fused_gather_attn_dsv4" in name:
        return "DSV4 gather attention"
    if "_fp8_paged_mqa_logits_kernel" in name:
        return "DSV4 full indexer Triton"
    if "_fp8_mqa_logits_post_kernel" in name:
        return "DSV4 indexer post Triton"
    if "qk_norm" in name or "rope" in name or "rotary" in name:
        return "DSV4 qk norm / rope"
    if "splitk" in name or "attn_reduce" in name or "decode_attention" in name:
        return "DSV4 splitk/reduce"
    if "ncclDevKernel" in name or "rccl" in name.lower():
        return "NCCL/RCCL kernel"
    if "reduce_kernel" in name:
        return "Torch reduce kernels"
    if (
        "elementwise" in name
        or "vectorized" in name
        or "BinaryFunctor" in name
        or "CUDAFunctor" in name
        or "launch_clamp" in name
    ):
        return "Torch elementwise kernels"
    if "gather_kernel" in name:
        return "Torch gather kernels"
    if "per_token_group" in name or "per_token_quant" in name or (
        "fp8" in name and "quant" in name.lower()
    ):
        return "FP8 per-token/group quant"
    if name.startswith("Cijk_") or "rocblas" in name.lower() or "gemm" in name.lower():
        return "CK/rocBLAS GEMM-like"
    return "other kernels"

stage_data = {}
trace_paths = glob.glob(os.path.join(trace_dir, "*.trace.json.gz"))
trace_paths += glob.glob(os.path.join(trace_dir, "cuda_graph_capture-*.json.gz"))
for path in sorted(trace_paths):
    stage = (
        "DECODE"
        if "-DECODE." in path or "cuda_graph_capture-" in os.path.basename(path)
        else "EXTEND"
    )
    rank = re.search(r"TP-(\d+)", os.path.basename(path)).group(1)
    with gzip.open(path, "rt") as f:
        events = json.load(f).get("traceEvents", [])
    cats = collections.Counter()
    calls = collections.Counter()
    steps = []
    graphs = []
    kernels = []
    timed_kernels = []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        dur_ms = event.get("dur", 0) / 1000.0
        if name.startswith("step["):
            steps.append((name, event.get("cat"), dur_ms))
        if name == "hipGraphLaunch":
            graphs.append(dur_ms)
        if event.get("cat") == "kernel":
            cat = category(name)
            cats[cat] += dur_ms
            calls[cat] += 1
            kernels.append((dur_ms, name))
            timed_kernels.append((event.get("ts", 0), dur_ms, cat, name))
    stage_data[(stage, rank)] = {
        "cats": cats,
        "calls": calls,
        "steps": steps,
        "graphs": graphs,
        "kernels": kernels,
        "timed_kernels": timed_kernels,
    }

if not stage_data:
    raise SystemExit(f"no traces found under {trace_dir}")

for stage in ("EXTEND", "DECODE"):
    ranks = sorted(rank for found_stage, rank in stage_data if found_stage == stage)
    if not ranks:
        continue
    print(f"\n=== {stage} ===")
    for rank in ranks:
        item = stage_data[(stage, rank)]
        print(
            "TP%s steps=%s graph_ms=%s"
            % (
                rank,
                [(name, cat, round(dur, 3)) for name, cat, dur in item["steps"][:4]],
                [round(dur, 3) for dur in item["graphs"][:4]],
            )
        )
    all_cats = sorted(
        set().union(*(stage_data[(stage, rank)]["cats"] for rank in ranks)),
        key=lambda cat: -sum(stage_data[(stage, rank)]["cats"][cat] for rank in ranks),
    )
    for cat in all_cats[:20]:
        vals = [stage_data[(stage, rank)]["cats"][cat] for rank in ranks]
        call_counts = [stage_data[(stage, rank)]["calls"][cat] for rank in ranks]
        print(
            f"{cat:30s} mean={statistics.mean(vals):8.2f} "
            f"min={min(vals):8.2f} max={max(vals):8.2f} calls={call_counts}"
        )
    print("top kernels TP0:")
    for dur, name in sorted(stage_data[(stage, "0")]["kernels"], reverse=True)[:12]:
        print(f"  {dur:8.3f} ms {name[:160]}")

    if stage != "DECODE":
        continue

    print("steady layer intervals (bounded by consecutive Mori dispatches):")
    intervals_by_rank = {}
    for rank in ranks:
        timed = sorted(stage_data[(stage, rank)]["timed_kernels"])
        boundaries = [
            ts
            for ts, _dur, cat, _name in timed
            if cat in ("Mori dispatch", "Mori AsyncLL dispatch send copy")
        ]
        intervals = []
        for begin, end in zip(boundaries, boundaries[1:]):
            wall_ms = (end - begin) / 1000.0
            per_cat = collections.Counter()
            for ts, dur_ms, cat, _name in timed:
                if begin <= ts < end:
                    per_cat[cat] += dur_ms
            intervals.append((wall_ms, per_cat))
        intervals_by_rank[rank] = intervals
        if not intervals:
            print(f"  TP{rank}: unavailable (fewer than two dispatch kernels)")
            continue

        median_ms = statistics.median(wall for wall, _cats in intervals)
        # Profiler startup can leave one rank waiting in its first collective.
        # Keep normal layer variation, but reject intervals far outside the
        # steady distribution and report how many were excluded.
        steady = [
            item
            for item in intervals
            if 0.25 * median_ms <= item[0] <= 3.0 * median_ms
        ]
        rejected = len(intervals) - len(steady)
        mean_wall = statistics.mean(wall for wall, _cats in steady)
        print(
            f"  TP{rank}: layers={len(steady)}/{len(intervals)} "
            f"mean_wall={mean_wall:.3f} ms median_wall={median_ms:.3f} ms "
            f"rejected_startup_or_gap={rejected}"
        )
        steady_cats = sorted(
            set().union(*(cats for _wall, cats in steady)),
            key=lambda cat: -statistics.mean(cats[cat] for _wall, cats in steady),
        )
        for cat in steady_cats[:12]:
            mean_cat = statistics.mean(cats[cat] for _wall, cats in steady)
            print(
                f"    {cat:30s} {mean_cat:7.4f} ms/layer "
                f"{100.0 * mean_cat / mean_wall:5.1f}% wall"
            )

    shared_cats = (
        "shared bounded gate/up GEMM",
        "dense W8A8 FP8 matmul",
    )
    print("shared-expert owner balance:")
    for rank in ranks:
        intervals = intervals_by_rank.get(rank, [])
        owner = [
            item
            for item in intervals
            if any(item[1][cat] > 0 for cat in shared_cats)
        ]
        other = [
            item
            for item in intervals
            if all(item[1][cat] == 0 for cat in shared_cats)
        ]
        if not owner or not other:
            print(
                f"  TP{rank}: owner_layers={len(owner)} other_layers={len(other)}"
            )
            continue
        print(
            f"  TP{rank}: owner_layers={len(owner)} other_layers={len(other)} "
            f"owner_wall={statistics.mean(x[0] for x in owner):.4f} ms "
            f"other_wall={statistics.mean(x[0] for x in other):.4f} ms "
            "shared_gate_up="
            f"{statistics.mean(sum(x[1][cat] for cat in shared_cats) for x in owner):.4f} ms"
        )

    if intervals_by_rank and all(intervals_by_rank.get(rank) for rank in ranks):
        num_layers = min(len(intervals_by_rank[rank]) for rank in ranks)
        critical_counts = collections.Counter()
        critical_walls = []
        for layer_idx in range(num_layers):
            layer_walls = {
                rank: intervals_by_rank[rank][layer_idx][0] for rank in ranks
            }
            critical_rank = max(layer_walls, key=layer_walls.get)
            critical_counts[critical_rank] += 1
            critical_walls.append(layer_walls[critical_rank])
        print(
            "  critical_rank_layers="
            f"{dict(sorted(critical_counts.items()))} "
            f"mean={statistics.mean(critical_walls):.4f} ms "
            f"p50={statistics.median(critical_walls):.4f} ms "
            f"max={max(critical_walls):.4f} ms"
        )
PY
}

case "${1:-}" in
  start)
    start_server
    ;;
  serve)
    serve_server
    ;;
  stop)
    stop_server
    ;;
  status)
    status
    ;;
  logs)
    tail -n "${2:-120}" "${LOG_FILE}"
    ;;
  bench)
    shift
    bench "$@"
    ;;
  bench-concurrent)
    shift
    bench_concurrent "$@"
    ;;
  bench-context)
    shift
    bench_context "$@"
    ;;
  start-dspark)
    start_server
    ;;
  serve-dspark)
    serve_server
    ;;
  bench-dspark)
    shift
    bench_dspark_concurrent "${1:-256}" 1 "${2:-1}"
    ;;
  bench-dspark-concurrent)
    shift
    bench_dspark_concurrent "$@"
    ;;
  profile)
    shift
    profile "$@"
    ;;
  parse-profile)
    shift
    parse_profile "$@"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
