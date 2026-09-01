# DSpark FP32 draft-LM-head rejection and unified-KV startup fix (2026-09-01)

## Scope

- DeepSeek-V4-Flash original weights, TP4/EP1/no-A2A, gamma-three DSpark.
- Physical GCDs 4--7; `amd-smi process --general --sort-by-pid` reported no
  running GPU processes before launch.
- Frozen 32-request heterogeneous workload SHA256:
  `f74de67a93a660cde060991df71c9e2972a05d82c3ba3f9fe7c144b1f066a152`.
- Existing accepted M128 attention multi-stream checkpoint remained enabled.

## Startup regression and fix

The first launch exposed an `UnboundLocalError` during low-tier CUDA/HIP graph
capture: the new DSpark M128 multi-stream selector read `unified_kv` before the
local variable was initialized.  Move the existing `is_unified_kv_triton()`
query before all multi-stream selectors.  This is an initialization-order fix;
it does not alter backend selection or native-AR math.

After the fix the service captured all 1--32 target and draft graph tiers.
Separate BS1 France validation passed the historical first-nine-token oracle
and semantic Paris check in 3/3 rounds.

## FP32 draft head screen

`SGLANG_DSPARK_FP32_LM_HEAD=1` was tested as the only acceptance-side candidate
not already covered by prior records.  It changes only the DSpark draft base
LM-head accumulation; native AR is unreachable.

Real heterogeneous BS32, 1024 tokens/request, `stream_interval=1`:

| round | resident tok/s | mean accepted length | all 32 length/finish |
|---:|---:|---:|:---|
| 1 | 1611.24 | 3.56843 | 1024 / `length` |
| 2 | 1624.83 | 3.60944 | 1024 / `length` |

The accepted BF16-draft-head checkpoint remains about 1646.59 tok/s median
(`1646.59/1648.25/1621.66`).  FP32 slightly raises accepted length but adds
enough draft cost that E2E does not improve.  Concurrent France remained
trajectory-sensitive (semantic in the first round, false in the second),
while the required separate BS1 oracle was exact.

## Decision

- Reject `SGLANG_DSPARK_FP32_LM_HEAD=1`; do not enable it in the launcher.
- Keep the `unified_kv` initialization-order fix.
- Current accepted E2E remains approximately 1646.6 tok/s resident BS32.

Artifacts:

- `/tmp/dsv4_fp32_lm_head_france.json`
- `/tmp/dsv4_fp32_lm_head_bs32_r1.json`
- the second round was emitted by the harness before its aggregate output file
  was finalized; the captured round values are recorded above.
