#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SGLANG_DIR:-/home/pc/Code/sglang}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/pc/models/modelscope}"
LANE="${LANE:-default}"
if [[ "${1:-}" == "batch" || "${1:-}" == "single" ]]; then
  LANE="$1"
  shift
fi

DEFAULT_GPUS="0,1,2,3"
DEFAULT_PORT="30000"
DEFAULT_BENCH_NREQ="4"
DEFAULT_CUDA_GRAPH_MAX_BS_DECODE="4"
DEFAULT_CHUNKED_PREFILL_SIZE="256"
DEFAULT_MORI_MAX_DISPATCH_TOKENS_PER_RANK="256"
DEFAULT_DISABLE_ATTN_TP_GATHER="0"
case "${LANE}" in
  batch|default)
    LANE_NAME="batch"
    ;;
  single)
    LANE_NAME="single"
    DEFAULT_GPUS="4,5,6,7"
    DEFAULT_PORT="30001"
    DEFAULT_BENCH_NREQ="1"
    DEFAULT_CHUNKED_PREFILL_SIZE="256"
    DEFAULT_MORI_MAX_DISPATCH_TOKENS_PER_RANK="256"
    DEFAULT_DISABLE_ATTN_TP_GATHER="1"
    ;;
  *)
    echo "unknown lane: ${LANE}" >&2
    exit 2
    ;;
esac

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-${DEFAULT_PORT}}"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="${LOG_FILE:-/tmp/sglang_dsv4_flash_${LANE_NAME}.log}"
PID_FILE="${PID_FILE:-/tmp/sglang_dsv4_flash_${LANE_NAME}.pid}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/sglang_speed_profile_dsv4_${LANE_NAME}}"
PROFILE_ID="${PROFILE_ID:-dsv4_${LANE_NAME}_probe}"

export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/python:${ROOT_DIR}/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:${ROOT_DIR}/python/sglang/kernels/aot/python}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm/core-7.14}"
if [[ -d "${ROCM_ROOT}/include" ]]; then
  export CPATH="${ROCM_ROOT}/include${CPATH:+:${CPATH}}"
fi
if [[ -d "${ROCM_ROOT}/lib" ]]; then
  export LIBRARY_PATH="${ROCM_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${ROCM_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${DEFAULT_GPUS}}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_HACK_FLASHMLA_BACKEND="${SGLANG_HACK_FLASHMLA_BACKEND:-triton}"
export SGLANG_OPT_USE_AITER_MHC_PRE="${SGLANG_OPT_USE_AITER_MHC_PRE:-0}"
export SGLANG_OPT_USE_AITER_MHC_POST="${SGLANG_OPT_USE_AITER_MHC_POST:-0}"
export SGLANG_OPT_USE_TRITON_MHC_COMBINE="${SGLANG_OPT_USE_TRITON_MHC_COMBINE:-1}"
export SGLANG_OPT_USE_TRITON_INDEXER_POST="${SGLANG_OPT_USE_TRITON_INDEXER_POST:-1}"
export SGLANG_OPT_USE_TRITON_INDEXER_FULL="${SGLANG_OPT_USE_TRITON_INDEXER_FULL:-1}"
export MORI_DISABLE_TOPO="${MORI_DISABLE_TOPO:-1}"
export MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-1}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-6G}"
export SGLANG_MORI_DISPATCH_DTYPE="${SGLANG_MORI_DISPATCH_DTYPE:-bf16}"
export SGLANG_MORI_COMBINE_DTYPE="${SGLANG_MORI_COMBINE_DTYPE:-bf16}"
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-${DEFAULT_MORI_MAX_DISPATCH_TOKENS_PER_RANK}}"

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
  --max-total-tokens "${MAX_TOTAL_TOKENS:-8192}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.3}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.82}"
  --disable-overlap-schedule
  --skip-server-warmup
  --enable-tokenizer-batch-encode
  --decode-log-interval "${DECODE_LOG_INTERVAL:-1}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
)
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-1}" == "1" ]]; then
  server_args+=(--disable-custom-all-reduce)
fi
if [[ "${DISABLE_ATTN_TP_GATHER:-${DEFAULT_DISABLE_ATTN_TP_GATHER}}" == "1" ]]; then
  server_args+=(--disable-attn-tp-gather)
fi
if [[ -n "${SPECULATIVE_ALGORITHM:-}" ]]; then
  server_args+=(--speculative-algorithm "${SPECULATIVE_ALGORITHM}")
fi
if [[ -n "${SPECULATIVE_DRAFT_MODEL_PATH:-}" ]]; then
  server_args+=(--speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}")
fi
if [[ -n "${SPECULATIVE_DSPARK_BLOCK_SIZE:-}" ]]; then
  server_args+=(--speculative-dspark-block-size "${SPECULATIVE_DSPARK_BLOCK_SIZE}")
fi
if [[ -n "${SPECULATIVE_DSPARK_SPS_TABLE_PATH:-}" ]]; then
  server_args+=(--speculative-dspark-sps-table-path "${SPECULATIVE_DSPARK_SPS_TABLE_PATH}")
fi
if [[ -n "${SPECULATIVE_DSPARK_CONFIDENCE_STS_PATH:-}" ]]; then
  server_args+=(
    --speculative-dspark-confidence-sts-path
    "${SPECULATIVE_DSPARK_CONFIDENCE_STS_PATH}"
  )
fi
if [[ "${SPECULATIVE_DSPARK_ALIGN_VERIFY_TOKENS_TO_GRAPH_TIER:-0}" == "1" ]]; then
  server_args+=(--speculative-dspark-align-verify-tokens-to-graph-tier)
fi

usage() {
  cat <<'EOF'
Usage: scripts/rocm_dsv4_flash.sh [batch|single] <command> [args]

Lanes:
  batch                 Defaults to GPU 0-3, port 30000, bench nreq=4.
  single                Defaults to GPU 4-7, port 30001, bench nreq=1.

Commands:
  start                 Start the TP4/EP4 ROCm DSV4 service in background.
  serve                 Run the TP4/EP4 ROCm DSV4 service in foreground.
  stop                  Stop the background service from the pid file.
  status                Show service process and ROCm VRAM/PID state.
  logs [n]              Tail the last n log lines, default 120.
  bench [tokens] [n] [reps]
                        Run official-prompt throughput probe.
                        Defaults: tokens=256, n=batch/single default, reps=1.
  both-bench [tokens] [reps]
                        Run batch lane and single lane probes concurrently.
                        Defaults: tokens=256, reps=1.
  profile [tokens] [n] [steps]
                        Start SGLang stage profiler, then send one request.
                        Defaults: tokens=32, n=batch/single default, steps=1.
                        ROCprofiler may segfault after stop on this stack.
  parse-profile [dir]   Summarize saved EXTEND/DECODE trace kernel time.

Optional env:
  SPECULATIVE_ALGORITHM=DSPARK
  SPECULATIVE_DSPARK_BLOCK_SIZE=5
  SPECULATIVE_DSPARK_SPS_TABLE_PATH=/path/to/sps.json
  SPECULATIVE_DSPARK_CONFIDENCE_STS_PATH=/path/to/sts.json
  DISABLE_ATTN_TP_GATHER=1   # single lane default; set 0 to restore padded graph capture.
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
      echo "ready: ${BASE_URL}"
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
  local nreq="${2:-${DEFAULT_BENCH_NREQ}}"
  local reps="${3:-1}"
  "${PYTHON_BIN}" - "${BASE_URL}" "${tokens}" "${nreq}" "${reps}" <<'PY'
import json
import sys
import time
import urllib.request

base_url, tokens, nreq, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
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
    sampling_params = {"temperature": 0, "max_new_tokens": tokens}
    if nreq == 1:
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "cache_salt": f"bench-{tokens}-{nreq}-{rep}-0-{salt_ns}",
        }
    else:
        payload = {
            "text": [prompt] * nreq,
            "sampling_params": sampling_params,
            "cache_salt": [
                f"bench-{tokens}-{nreq}-{rep}-{i}-{salt_ns}" for i in range(nreq)
            ],
        }
    print(f"BEGIN rep={rep} tokens={tokens} nreq={nreq}", flush=True)
    dt, out = send(payload, timeout=max(1200, tokens * max(2, nreq)))
    rows = out if isinstance(out, list) else [out]
    lens = [len(row.get("output_ids") or []) for row in rows]
    total = sum(lens)
    first_text = (rows[0].get("text") or "")[:96]
    reasons = [row.get("meta_info", {}).get("finish_reason") for row in rows]
    print(
        "END "
        f"rep={rep} wall={dt:.3f}s nreq={len(rows)} out_lens={lens} "
        f"total_tokens={total} sum_tok/s={total / dt:.3f} "
        f"per_req_tok/s={(total / len(rows)) / dt:.3f} "
        f"decode_step_ms={(dt / max(lens or [1])) * 1000:.1f} "
        f"finish={reasons} "
        f"text0={first_text!r}",
        flush=True,
    )
PY
}

both_bench() {
  local tokens="${1:-256}"
  local reps="${2:-1}"
  local script_path
  script_path="$(realpath "$0")"
  "${script_path}" batch bench "${tokens}" 4 "${reps}" &
  local batch_pid=$!
  "${script_path}" single bench "${tokens}" 1 "${reps}" &
  local single_pid=$!
  local rc=0
  wait "${batch_pid}" || rc=$?
  wait "${single_pid}" || rc=$?
  return "${rc}"
}

profile() {
  local tokens="${1:-32}"
  local nreq="${2:-${DEFAULT_BENCH_NREQ}}"
  local steps="${3:-1}"
  rm -rf "${PROFILE_DIR}"
  mkdir -p "${PROFILE_DIR}"
  curl --noproxy '*' -sS --max-time 30 -X POST "${BASE_URL}/start_profile" \
    -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"${PROFILE_DIR}\",\"profile_id\":\"${PROFILE_ID}\",\"num_steps\":${steps},\"profile_by_stage\":true,\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false}"
  echo
  bench "${tokens}" "${nreq}" 1
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
for path in sorted(glob.glob(os.path.join(trace_dir, "*.trace.json.gz"))):
    stage = "DECODE" if "-DECODE." in path else "EXTEND"
    rank = re.search(r"TP-(\d+)", os.path.basename(path)).group(1)
    with gzip.open(path, "rt") as f:
        events = json.load(f).get("traceEvents", [])
    cats = collections.Counter()
    calls = collections.Counter()
    steps = []
    graphs = []
    kernels = []
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
    stage_data[(stage, rank)] = {
        "cats": cats,
        "calls": calls,
        "steps": steps,
        "graphs": graphs,
        "kernels": kernels,
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
  both-bench)
    shift
    both_bench "$@"
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
