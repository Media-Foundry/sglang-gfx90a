# TP8 small-prefill runtime-M repair — 2026-09-08

## Decision and scope

Accept and enable `SGLANG_DSV4_GFX90A_FP4_RUNTIME_M` by default, with a narrow
runner guard: native (no speculative algorithm), TP8 expert weight shapes,
129 <= actual rows <= 1023, generic grouped route only (not MFMA32 prefill).
Setting the flag to 0 restores exact-M specialization. Original checkpoint,
quantization, sorting, arithmetic and fixed-slot reduction order are unchanged.
TP4, speculative paths, captured decode tiers <=128 and large CK/MFMA routes
remain unchanged. This is a row-count guard, not a forward-mode assertion;
hypothetical native eager decode with >128 rows can also use runtime M.
Rows <=128 still specialize; this is not a universal cold-JIT solution.

## Root cause and implementation

See `dsv4_tp8_c32_ar_jit_wait_20260908.md`. Real C32 coding requests formed
different cap-16 prefill row sums on each wave. Gate/down embedded exact M in
both the JIT key and C++ template, causing four serialized compilations for
two prefill shapes. A traced M369/M368 wave spent 24.424 seconds compiling.
Seven ranks correctly waited on the compiler owner's file lock. Do not remove
locks, erase caches or blame useful GPU prefill math for this interval.

Use template M=0 as a shared runtime-M specialization. Host tensor checks and
launch bounds use the actual tensor row count; grouped gate/down kernels take
M as a scalar. Static templates retain their original compile-time M. The
runtime reducer preserves `(slot0+slot4)+(slot1+slot5)+slot2+slot3` order.
There is no padding/bucket approximation and no changed weight layout.

Files: `gfx90a_fp4_expert_gemv.cuh`, corresponding ops wrapper, AIter runner,
`environ.py`; oracle `scripts/rocm/check_dsv4_fp4_runtime_m.py`, CPU guards
`test/registered/unit/model_executor/runner/test_dsv4_fp4_runtime_m.py`.

## Validation

- Actual TP8 shapes, M129/369/511/1023: static/runtime full routed stage,
  gate intermediate and final BF16 output bitwise exact; finite outputs.
- Per shape: 10 input/scale/route-weight mutations, 10 graph replay checks per
  mutation. Full-stage ABBA timings (static/runtime, microseconds):
  1124.127/1123.207, 2658.910/2655.518, 3492.492/3489.013,
  6660.603/6658.419. Essentially unchanged GPU compute cost, not a kernel speedup.
- CPU loader-key/runner guards plus existing TP8 wo_a guard tests: 5 passed.
- France exact. C1 nine measured 256-token completions retained the baseline
  hashes. Three case medians 84.112/84.141/84.123 tok/s (baseline ~84.17).
  C1 corpus is mixed-purpose, not exclusively coding requests.
- Six fixed-prefix probes retained next IDs, input logprobs and output top-20
  logprobs exactly. After C32, 60 additional fresh-cache fixed-prefix transition
  probes passed all three exact checks. These are prefill probes, not a full
  cached-decode versus recompute proof.

## C32 end-to-end result

Native TP8/EP1/no-A2A, GPUs0–7, pool131072, memory0.80, graph1/2/4/8/16/24/32,
overlap/SBO enabled, prefill profile M36864 with cap16 and 20ms delayer.
32 distinct real coding prompts, same selected token-ID manifest as baseline,
737 total input tokens; six waves, 256 forced output tokens each, temperature0,
stream interval1, fresh salts and start barrier. No speculative decoding.

| Metric | Exact-M baseline | Runtime M |
|---|---:|---:|
| Complete-request median tok/s | 240.216 | 922.033 |
| Complete-request trimmed mean tok/s | see JSON | 921.879 |
| Resident decode-window median tok/s | 959.878 | 963.292 |
| Runtime-M complete wave seconds | — | 8.8847 median |

Runtime-M six complete-request rates: 918.820, 920.854, 922.018, 922.048,
922.597, 923.502 tok/s. All 192 requests completed 256 tokens / finish=length.
This is removal of repeated cold-shape JIT latency, NOT a 3.84x decode-kernel
speedup. E2E comparison is sequential before/after, not ABBA. Shared runtime
modules had already been loaded by correctness probes; first-ever module
compilation and startup rebuild are not included in these six waves.
Small exact shapes <=128 still caused cold compiles during C1 warmup after the
header change; no claim of zero cold-start latency.

Read-only 90-second process observer covering C32 saw all eight ranks, zero
compiler samples and zero JIT-lock samples. `/proc/2620885/maps` confirmed loaded
`gate_up_grouped_256_0_6_256_4096_4_2_8_416_0` and
`down_grouped_256_0_6_4096_256_4_2_8_312_0` modules.

Both baseline and candidate have 0/32 full completion hashes identical across
all six C32 waves. This existing batch-dependent long-trajectory variability
is NOT fixed or certified by this change. Exact component tests and fixed-prefix
tests are the acceptance evidence; normal completion alone is not a semantic
correctness proof. No generated code was executed.

## Artifacts and handoff

Adjacent JSON preserves oracle results, per-wave timings, completion hashes,
workload hashes and exact-check/observer summaries. Original detailed artifacts
are listed by filename there under `/tmp`. Server log:
`/tmp/dsv4_tp8_runtime_m_B_20260908.log`; parent2620634, localhost30011.
Service explicitly uses runtime flag1 (same effective setting as new default),
TP8_BS1_WOA_GEMV=1, PREFILL_THROUGHPUT_PROFILE=1,
TP8_MULTI_REQUEST_PROFILE=1, C4_TRIVIAL_LOGITS_SKIP=1. Service remains running.

Do not extend to TP4, speculative kernels, row-prefetch/DPP variants or larger
rows without separate exactness/performance validation. Further work can remove
other exact-M compile keys, but must preserve existing decode geometry contracts.
