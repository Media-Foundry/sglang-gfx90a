# Original V4 C16 prefill: skip causal empty indexer tiles

Independent follow-up after the compressor drift ablation. No new weight dtype,
no query selection approximation, no cache truncation. Original AR selector is
unchanged. New default-off `SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP=1` only allows
ordinary original-V4 EXTEND, unified pool, non-draft/non-DSpark/non-MTP backend,
canonical SGL Top512. After the completed ABBA below, the launcher opts in only
when BOTH TP8 multi-request and large-prefill profiles are enabled, with
TP8/EP1/no-A2A. Explicit0 is preserved; the runtime flag otherwise defaults off.
It selects the existing nonempty kernel via the public
wrapper: nonempty tiles execute the original kernel; empty tiles write zero.
Query producers, compressor/cache updates and true per-row lengths are unchanged.

## Component gate (physical GPU4, no service concurrently)

`scripts/rocm/check_dsv4_indexer_empty_tiles.py --production --prefill`
models zero-prefix8K requests with per-request independent pages and causal C4
lengths, width2048, trivial_topk512, BLOCK_S16. Synthetic Q/K/weights, not real
service capture. Timing includes logits and deterministic logical/physical
Top-K, **not query projections or the full indexer**.

100 Q/K/weights/length/page mutations and100 fixed graph replays per shape;
boundary/tie fixtures; all output score bits and both index arrays exactly equal.
Three ABBA cycles,5 graph iterations per timing observation:

| M | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|
|8192|11.427512|7.921278|1.442635x|
|32768|45.640773|31.637561|1.442614x|

Elapsed component time decreases about30.68%, **not an E2E measurement**.
CPU tests8/8 include independent opt-in, fail-closed model/worker roles,
non-EXTEND rejection and unchanged AR/public-wrapper contracts.

Evidence: `.agents/experiments/dsv4_input_identity_20260914/`
`prefill-empty-tile-{screen,full}.json`. First standalone invocation failed at
import (missing local `sgl_kernel` PYTHONPATH); no GPU test ran in that attempt.
Successful commands use the same three PYTHONPATH entries as launcher:
`python`, `python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312`,
`python/sglang/kernels/aot/python`, plus HIP_VISIBLE_DEVICES=4, SGLANG_USE_AITER=1.

## Service protocol

`.agents/experiments/dsv4_tp8_c16_empty_tiles_20260914/sweep.py` runs fresh A1,
B(B1/B2),A2 processes. Each leg has3 warmed waves, same16 diverse real-code
requests /131069 input tokens, native TP8/EP1, chunk32768,1M logical KV,
decode graphs1/2/4/8/16/32/64. All stability and capture diagnostics are
explicitly disabled. Backend actual-hit assertion follows warmup. France and
two128-token quality/input-echo/ID-text checks per process are separate from
timed P (one generated token).

### Rejected incomplete B attempt (harness failure, retained)

First A1 completed at5295.009 tok/s. First B timed legs were5571.646/5570.429,
but its subsequent quality phase failed before requests: a newly prepared local
`profile.py` shadowed stdlib `profile` when transformers imported cProfile.
A direct CPU import reproduced the exact traceback. This was an agent-created
tooling error, not a GPU/kernel failure. PID1128695 was stopped with no remaining
owned processes. Files are retained in `B-import-shadow-failed/` and the matching
run log; those timings are excluded from the final acceptance table.

Renamed the tool to `capture_timeline.py`, added stdlib-name collision and import
preflight tests, and moved tokenizer imports before service startup. Resume from
B reruns the whole candidate arm (warmup, six timed waves, two quality waves),
then runs A2. Preserve the already completed A1; model/runtime sources unchanged.

## Completed accepted ABBA

| Leg | Three warmed input tok/s measurements | Median |
|---|---|---:|
| A1 |5295.009 /5301.482 /5292.931|5295.009|
| B1 |5570.699 /5566.164 /5569.046|5569.046|
| B2 |5570.901 /5570.522 /5566.221|5570.522|
| A2 |5299.479 /5304.543 /5301.296|5301.296|

Mean of control leg medians **5298.152392**, candidate **5569.783981 input
tok/s**, **+5.126912%**. Mean leg median request TTFT **15.551403 ->14.806708s**,
**-4.788604%**. All four legs have identical logged shape counts:12 forwards
of4 requests/M32768 per3 waves (4 forwards per wave). Runtime indexer first-hit
logs in all8 candidate ranks show actual query rows32766, C4 capacity2048,
BLOCK_S16; scheduler shape reporting includes alignment and is not a proof
every actual query tensor has32768 rows.

All96 long-output request input echoes were exact, zero prefix hits,128
completion tokens with ID/text consistency. All3 France sentinels answered
Paris. Within-process repeated128-token output equality: A1 15/16, B16/16,
A2 15/16. Candidate excerpts are coherent code-analysis responses, truncated
at128 tokens; this is not an executable-code oracle or whole-model determinism.
The exact component comparison establishes that the changed indexer operation
preserves score bits and logical/physical selection. Do not attribute the
improved repeat count to this optimization.

Each fresh service completed and stopped with no remaining owned processes;
AMD-SMI confirmed all8GCDs free after A2. Runtime source hashes match across all
accepted arms. The failed earlier B is archived but excluded from the summary.

Decision: retain and default-enable in the matched TP8 large-prefill profile.
No new performance claim for C1,64K chunks, DSpark or decode. No additional
KV/weight workspace introduced. Full critical-path profiling is next, separately
from these accepted throughput numbers.
