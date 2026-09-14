# Original V4 C16 prefill: skip causal empty indexer tiles

Independent follow-up after the compressor drift ablation. No new weight dtype,
no query selection approximation, no cache truncation. Original AR selector is
unchanged. New default-off `SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP=1` only allows
ordinary original-V4 EXTEND, unified pool, non-draft/non-DSpark/non-MTP backend,
canonical SGL Top512. It selects the existing nonempty kernel via the public
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

## Service acceptance, not yet established

`.agents/experiments/dsv4_tp8_c16_empty_tiles_20260914/sweep.py` runs fresh A1,
B(B1/B2),A2 processes. Each leg has3 warmed waves, same16 diverse real-code
requests /131069 input tokens, native TP8/EP1, chunk32768,1M logical KV,
decode graphs1/2/4/8/16/32/64. All stability and capture diagnostics are
explicitly disabled. Backend actual-hit assertion follows warmup. France and
two128-token quality/input-echo/ID-text checks per process are separate from
timed P (one generated token). No default-on or speed claim until this finishes.

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
