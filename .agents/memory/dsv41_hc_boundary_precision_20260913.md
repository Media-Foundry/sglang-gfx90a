# V4.1 HC / Engram reduction precision, 2026-09-13 evening

User resumed after the emergency job. Starting HEAD `44b40609b0`; amd-smi
reported all eight GCDs empty. Original V4.1 weights, TP8/EP1/no-A2A, eager
native AR, chunk2304, 32768 pool/context, pinned host Engram. This is a
correctness investigation, not a performance benchmark. Unrelated dirty
`cuda_graph_runner_memory_usage.pickle` and old untracked files are preserved.

## Service-reproducing evidence

Baseline diagnostic PID420060 ready21:50:59; 104.04s weight load. Only the two
previous opt-ins (`WO_A_INVARIANT`, `ATTN_INVARIANT`) enabled. Same SQL203 IDs,
two generated IDs `[666,31151]`, then fixed204 recompute. It reproduces prior
max common Top20 logprob delta **0.37497711181640625** exactly.

An explicit `--forward-hooks` factory wraps HC methods and Engram.apply_gate
only for bounded diagnostic capture. Originals run once; output identity is
preserved. Snapshots before/after calls synchronize CPU/GPU and invalidate
timing or absence-of-race claims. No parameter, cache or collective is replaced.
HC layers12/13/20/21/22, rank0, two sublayers per forward; Engram both modules.

All **22 component replays reproduce both recorded service outputs**: ten HC
coefficient boundaries, ten HC posts, two Engram gates. Identical-input
M1/M204 probes isolate shape effects rather than propagation from earlier
different inputs:

- Layer13 attention HC input and collapsed/normalized input agree, but pre,
  post and comb differ by up to 5.96e-8 / 2.24e-8 / 7.45e-8. The following
  FFN consumes the differing pre coefficients and gets the single BF16
  difference observed in the previous run. HC post is not the source here.
- Captured HC FP32 dot products agree at M1/M204. RMS normalization does not:
  layer13 rstd max difference **4.76837158203125e-7**. Layer20/21 examples also
  reproduce shape-dependent rstd differences.
- Both Engram x and projected KV agree. Second Engram gate's raw dot differs
  by **0.0001220703125**, normalized dot by3.814697265625e-6, gate by
  7.450580596923828e-9; final BF16 output differs in one element by
  1.1920928955078125e-7. The first Engram has internal reduction differences
  too, but its gate/output round to equal values for this fixture.
- All ten HC post replays on identical repeated inputs are equal across M.

Correction to previous memory wording: the second Engram is configured at
**layer14**, not layer13. It executes after the preceding layer13 boundary.
The older module-number-only hook did not store an Engram layer_id. The
config's actual Engram layer IDs are `[1,14]`; do not infer layer ID merely
from adjacent norm/attention entries in a trace.

## Narrow fixes and the failed intermediate candidate

1. Tried fixed RMS only while retaining ordinary HC FP32 GEMM. This fixes
   the source visible in the recorded M1/M204 pair, but the expanded mixed-row
   test fails on layer12 attention, M64, row32. Ordinary GEMM is therefore not
   batch-invariant in general either. That intermediate selector was never
   used in a service or committed as an accepted fix.
2. `SGLANG_DSV41_HC_STATS_INVARIANT` (default **False**) now fixes both RMS
   and dot. RMS is a fixed one-row reduction; dot reuses the existing
   `_gfx90a_mhc_mix_kernel`, N4/K256, four waves, FP32 products/accumulation,
   no FP fusion. It does not expand/cache checkpoint weights or reduce precision.
3. `SGLANG_DSV41_ENGRAM_GATE_INVARIANT` (default **False**) enables the
   already-existing `fused_engram_gate` on HIP via Engram.apply_gate. It uses
   fixed per-row reductions and one BF16 output cast. Host lookup/storage and
   wkv projection are unchanged. Original V4 Flash selectors are untouched.

Real-fixture validation: ten HC coefficient inputs plus two Engram gates,
each with 32 first/middle/last placement checks at M2/3/15/16/17/32/64/128/
203/204/256, with surrounding rows perturbed. **384/384 pass**. Each component
also passes **100 input mutations** and **1000 individually compared HIP Graph
replays**, with in-place input changes every ten replays. Only these components
are graph-validated, not the full serving path.

Independent FP64 references are not bitwise equal to all candidate outputs:
HC max relative L2 about2.55e-7; first Engram has two BF16 differences,
max6.103515625e-5, relative L2 about7.66e-6; second Engram is equal. Explicit
reference tolerances in the oracle are HC atol5e-6/rtol2e-5, Engram
atol1e-6/rtol0.015625; these are not whole-model acceptance thresholds.

CPU suite passed **127 tests +16 subtests** with GPU4 visible for import-time
architecture queries. An initial attempt to hide all GPUs failed during
collection because existing ROCm imports query device0; no tests ran in that
attempt. No driver/tool failure is claimed.

## Artifacts and handoff

Reports and reproduction launchers: `.agents/experiments/dsv41_hc_boundary_20260913/`.
Raw tensor captures remain `/tmp/dsv41-hc-boundary-20260913/` (not Git).
GPU components used physical GPU4, while the diagnostic service was idle.

Baseline PID420060 was shut down after requests finished; all GPUs confirmed
empty before starting the combined candidate process. Full-model acceptance
is pending below. Do not enable either new selector by default based only on
these component results.

## Combined candidate: first-decode discrepancy eliminated in this probe

New process PID433583, ready22:00:13, 103.60s weight load. Both previous
attention/wo_a opt-ins plus the new HC_STATS and ENGRAM_GATE opt-ins enabled.
Original checkpoint precision and host Engram table path unchanged.

For the fixed204 prefix, two independent fresh-cache recomputes have identical
committed ID31151, identical Top20 membership/order and **zero common logprob
delta**, versus0.37497711181640625 before the two reduction fixes. The recorded
first two continuation IDs remain `[666,31151]`.

Full module trace comparison at cached row0 / prefill row203 finds **164/164
captured output tensors exact**, including the final RMSNorm. No nonfinite
output found. This is rank0's selected-token observation, not a full-vocabulary
logit capture, all-rank trace, arbitrary-batch guarantee or whole-model graph
test. The independent longer natural-EOS SQL/fixed-prefix run is in progress.
