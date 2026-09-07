# Speed verification after native numerical drift fixes

## Question and comparison boundary

User asked whether drift repairs cost speed. Runtime code is current main
`9967662469` plus unchanged pre-existing local experiments. No weight or
inference algorithm optimization is added in this measurement turn.

The isolated ABBA compares **A: legacy atomic Top-K (mode 0)** with
**B: deterministic HIP Top-K (mode 2)**. Both retain the fixed MFMA wave
shuffle and corrected preshuffled indexer cache reader. A is explicitly
not a correct deployment candidate. This does not reconstruct all old
bugs, nor claim to measure the combined cost of reverting all three fixes.

Four independent TP4/EP1/no-A2A services, physical GCDs 4–7, order A1/B1/B2/A2.
Both arms: chunk2304, pool65536, graph BS1, memory fraction .80, native AR,
MFMA32/64 enabled, **scheduler overlap disabled**. Before each startup:
AMD-SMI process/resource snapshot. GC frozen once and each measured shape
warmed. The controller stops only its own services; it refuses an occupied
port or overwriting an existing experiment directory.

## Workloads and units

- C1: exact historical three code prompts (Python linked list, SQL duplicate
  emails, merge sorted arrays), 256-token greedy completion, non-streaming
  HTTP wall time. One excluded warmup per case then three measured rounds.
  France sentinel and six fixed-continuation logprob probes accompany each
  service. Short prompts do not exercise the long-context Top-K path.
- C1 prefill: fixed real 2304-token source prompt at manifest index2, one
  generated token, five rounds, first excluded. Report streaming TTFT.
- C16 prefill: 16 distinct real 2304-token source prompts, simultaneous
  independent HTTP submissions, one output token, three rounds, first
  excluded. Report 36864 input tokens divided by group first-token wall
  time. These are M2304 GPU chunks, not simultaneous M36864 GPU prefill
  and not concurrent decode output throughput.
- Every timing request has a fresh cache salt. C1 harness now records and
  asserts cached_tokens=0. Hashes, outputs and raw timings are saved.

## TP8 fixed-path observation before TP4 ABBA

Same graph/no-overlap validation profile on one TP8 model, GCDs0–7:

- C1 per-case medians: **60.717 /60.755 /60.670 HTTP tok/s**.
- All-nine measured-request median: **60.680 tok/s**; France passes.
- C16 prefill group TTFT: 10.092 /10.117 /10.130 seconds; excluding the
  first round gives **3641.38 input tok/s** median; first-token outputs exact.

These TP8 numbers must not be compared with historical TP4 C1 75 tok/s as
if TP size were unchanged. Historical TP4 75 also had scheduler overlap
enabled and a smaller pool. The no-overlap A1 legacy-Top-K control currently
measures only 65–67 tok/s, so the historical gap cannot simply be attributed
to the new Top-K. A separate overlap-enabled repeat is planned.

## Artifacts and status

Controller: `scripts/rocm/run_dsv4_topk_speed_abba.py`.
Analysis: `scripts/rocm/summarize_dsv4_topk_speed_abba.py`.
Raw ABBA: `/tmp/dsv4_topk_speed_abba_20260907/`.
TP8: `/tmp/dsv4_speed_tp8_fixed_c1.json` and
`/tmp/dsv4_speed_tp8_fixed_prefill_c16.json`.

## Completed TP4 ABBA results

| Metric (warm samples only) | A legacy Top-K | B deterministic Top-K | B versus A |
|---|---:|---:|---:|
| C1 HTTP output tok/s, median of 18 | 66.422 | 66.341 | -0.123% |
| C1 HTTP output tok/s, trimmed mean | 66.346 | 65.820 | -0.793% |
| 2304-token C1 TTFT, median of 8 | 0.933107 s | 0.937578 s | +0.479% |
| C16 prefill input tok/s, median of 4 | 2479.669 | 2464.855 | -0.597% |

The measured differences are below 1%, not evidence of a material regression
on these workloads. Do not claim universal zero overhead or a statistically
proven speedup. B2's C1 samples range down to 58.67 tok/s; B overall range
58.67–68.20 shows runtime jitter, so report the aggregate rather than one arm.

All three B C1 output-ID sequences agree across B1/B2 and their warmups.
All six fixed-continuation probes match input logprobs and output top-20
exactly across the two B processes. B prefill first-token outputs are stable;
A C16 first-token outputs are not consistently stable. A remains diagnostic.

## Restoring scheduler overlap

One additional fixed-path TP4 service changes only
`DISABLE_OVERLAP_SCHEDULE=1` to `0` relative to the ABBA B settings. It retains
pool65536, graph BS1, chunk2304, MFMA32/64, and deterministic Top-K.

- Per-case HTTP medians: **76.886 /76.839 /76.180 tok/s**.
- All-nine median **76.839 tok/s**, trimmed mean **76.424 tok/s**.
- France passes. All six fixed-continuation input/output logprob probes
  exactly match no-overlap B1. Code completion hashes agree too.
- This recovers the historical ~75 C1 level without reverting the numerical
  fixes. The ~66 validation speed was mainly a scheduler-overlap configuration
  difference. This additional schedule comparison is not another full ABBA.

Longer C1 (real request index2, 2304 input tokens, 256 output tokens), five
rounds with first excluded: median full streaming HTTP **55.050 output tok/s**,
median post-first-token streaming decode window **68.685 tok/s**, median
TTFT **0.934510 s**. All five full completions match. Do not compare its full
HTTP rate directly with short-prompt C1: nearly a second is prefill.

## Important limitation: long-context Top-K component is slower

Additional synthetic single-row GPU-event ABBA, physical GPU0, fixed random
scores, 50 sequential nodes per graph and 20 replays per timing (Python
launch overhead amortized), gives the following pair means:

| Logical C4 candidates | Legacy Top-K | Deterministic Top-K |
|---:|---:|---:|
| 576 | 8.687 us | 35.591 us |
| 4096 | 11.042 us | 50.446 us |
| 16384 | 19.932 us | 125.748 us |

This is a genuine component cost, and differs from comparing against the
temporary PyTorch sort/gather repair (which the HIP kernel beat). The new
whole-row radix passes and deterministic emission are correctness-first.
Large-candidate, long-context decode may regress even though measured short
C1 and 2304-prefill E2E changes are below 1%. No 16K/64K-context E2E ABBA has
been completed, so do not translate these ratios into whole-model slowdowns.
Next optimization should preserve deterministic score/ID tie-breaking and
publication order while reducing scan/histogram work, not restore atomics
that arbitrarily choose tied entries.

Reproduction: `scripts/rocm/bench_dsv4_deterministic_topk_cost.py`.
Additional artifacts: `/tmp/dsv4_speed_tp4_fixed_overlap_c1.json` and
`/tmp/dsv4_speed_tp4_fixed_overlap_long_c1.json`.
