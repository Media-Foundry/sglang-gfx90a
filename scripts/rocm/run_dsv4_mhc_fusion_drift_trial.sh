#!/usr/bin/env bash
# Drive the four ABBA arms of the fused-MHC drift trial.
#
# Each arm gets a fresh service, because the candidate env is read during graph
# capture and cannot be flipped on a live service. `start-dspark` daemonizes and
# runs the launcher's own wait_ready (log marker + TCP connect + freeze_gc), so
# this script must not add an HTTP readiness poll of its own.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH="${ROOT}/scripts/rocm_dsv4_flash.sh"
OUT="${OUT_DIR:-/tmp/dsv4_tp8_mhc_fusion_drift_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-30021}"
BASE="http://127.0.0.1:${PORT}"
PYTHON_BIN="${PYTHON_BIN:-/home/pc/anaconda3/envs/DS/bin/python}"
TOKENS="${TOKENS:-512}"
WAVES="${WAVES:-2}"
mkdir -p "${OUT}"

# Per-arm log/pid paths so one arm cannot adopt another's service, and so
# `is_running` cannot silently turn a start into a no-op against a stale env.
arm_log() { echo "${OUT}/server_$1.log"; }
arm_pid() { echo "${OUT}/server_$1.pid"; }

require_free_gpus() {
  local held
  held="$(amd-smi process --general --sort-by-pid -g 0 1 2 3 4 5 6 7 2>/dev/null |
          grep -c 'PID:' || true)"
  if [[ "${held}" != "0" ]]; then
    echo "GPUs are busy (${held} process entries); refusing to start." >&2
    amd-smi process --general --sort-by-pid -g 0 1 2 3 4 5 6 7 2>/dev/null |
      grep -E 'GPU:|PID:|NAME:' | head -24 >&2
    exit 3
  fi
}

start_arm() {
  local arm="$1" fusion="$2"
  echo "=== arm ${arm}: SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION=${fusion} ==="
  SGLANG_DSV4_GFX90A_DSPARK_TP8_FULL_TARGET_PROFILE=1 \
  SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION="${fusion}" \
  PORT="${PORT}" \
  LOG_FILE="$(arm_log "${arm}")" \
  PID_FILE="$(arm_pid "${arm}")" \
    "${LAUNCH}" start-dspark
  # Confirm the arm's env actually reached the server process, rather than
  # trusting that the export propagated.
  local pid observed
  pid="$(cat "$(arm_pid "${arm}")")"
  observed="$(tr '\0' '\n' < "/proc/${pid}/environ" |
              grep '^SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION=' || true)"
  if [[ "${observed}" != "SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION=${fusion}" ]]; then
    echo "arm ${arm} env mismatch: got '${observed:-unset}'" >&2
    exit 5
  fi
  echo "  pid=${pid} env=${observed}"
}

stop_arm() {
  local arm="$1"
  LOG_FILE="$(arm_log "${arm}")" PID_FILE="$(arm_pid "${arm}")" "${LAUNCH}" stop || true
  sleep 20
}

require_free_gpus
for spec in "A1 0" "B1 1" "B2 1" "A2 0"; do
  set -- ${spec}
  arm="$1"; fusion="$2"
  trap 'stop_arm "${arm}"' EXIT
  start_arm "${arm}" "${fusion}"
  "${PYTHON_BIN}" "${ROOT}/scripts/rocm/bench_dsv4_tp8_mhc_fusion_drift_trial.py" \
    --base-url "${BASE}" --arm "${arm}" --tokens "${TOKENS}" --waves "${WAVES}" \
    --output "${OUT}/arm_${arm}.json" 2>&1 | tee -a "${OUT}/trial.log"
  stop_arm "${arm}"
  trap - EXIT
done

"${PYTHON_BIN}" "${ROOT}/scripts/rocm/analyze_dsv4_mhc_fusion_drift_trial.py" \
  --dir "${OUT}" --output "${OUT}/verdict.json" 2>&1 | tee -a "${OUT}/trial.log"
echo "artifacts: ${OUT}"
