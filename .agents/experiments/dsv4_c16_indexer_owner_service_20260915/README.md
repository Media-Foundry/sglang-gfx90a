# C16 indexer query-owner service experiment — accepted

Original V4, original checkpoint, TP8/EP1/native AR, 1M logical KV, 32K
prefill budget; accepted paired-column MHC/query16/runtime-M baseline.
New flag `SGLANG_DSV4_C4_PREFILL_QUERY_OWNER` remains default-off in direct
launches, and is enabled only by the combined TP8 multi-request and prefill
throughput launcher profile (explicit0 wins). Query projection, compressor/cache population, sparse
selection and AR decode remain unchanged.

The helper creates active original-query16-group row maps from existing CPU
prefix/extend metadata once per forward. It does not copy live GPU lengths to
CPU. The debug-only check compares these expected lengths to actual GPU C4
metadata and compares complete full-local vs owner scores and Top-K results.
Integer logical IDs are exchanged via existing TP device-group RCCL, then
mapped through each rank's local page table. No additional peer allocator.

## Completed gates

- Nine CPU tests passed (ownership metadata plus scope/MHC baseline tests).
- Eight-rank real layer20 integrated-helper test passed three input variants,
  differing rank-local physical page numbering and optional raw-ID output.
  `../dsv4_c16_indexer_owner_20260915/integrated.json`.
- `check/`: fresh actual service with dual-computation assertions enabled;
  16 real 8K code prompts, zero cache hits, exact echoed inputs, 128 tokens
  each; all outputs matched the previous accepted B/quality-0 answers exactly.
  France passed; short France did not enter owner scope. 1M KV remained.
  Process1600015 stopped cleanly. Diagnostic duration is NOT a speed result.
- Formal A1: three rates7373.586194/7377.219290/7373.833448 input tok/s,
  median7373.833448. Two128-token quality waves completed. Process1606045
  stopped cleanly. Warmup6506.534754 excluded from formal timing.

## Completed ABBA

Fresh-service A1→B1/B2→A2 completed, all owned processes stopped. All formal
legs use three waves, zero prefix hits, same frozen source/input manifest,
with dual-computation checks DISABLED. Raw first-token timestamps were
independently recomputed by `analyze.py` (192 exact formal input echoes).

| Leg | Median input tok/s |
|---|---:|
| A1 |7373.83345|
| B1 |7957.03975|
| B2 |7950.87246|
| A2 |7375.01017|

Control center7374.42181 → candidate7953.95610, **+7.85871%**. Metric is total
131069 new input tokens / earliest request start to latest first token, not
sum-of-per-user rates or decode throughput. Warmups excluded. Approximately
17.77→16.48s per C16x8K wave. No DSpark/accepted-token metric is involved.

Long-output comparison covers96 exact input echoes.31/32 candidate answers
match current controls; the remaining case8 text exactly matches an earlier
accepted control. `manual-review.md` records the bounded coherence review and
limits. `summary.json`/`quality-review.json` retain their automatic pending
review labels; the separate manual review and acceptance record close that
gate without rewriting raw analysis. Global numerical drift is NOT solved.

Additional eight-rank helper checks passed M8192/8193/12289/65536 with
captured values and varied prefix/extend metadata, plus all-trivial fallback.
These are ABI/metadata fixtures, not additional real-service performance runs.
An undersized score-width cap is now rejected before collective/Top-K work.
The tested helper is preserved byte-for-byte as `tested_helper.py`; the only
post-ABBA code change is that CPU metadata safety guard. AST equivalence and
GPU checks validate the unchanged supported arithmetic. Launcher promotion
only changes default flag resolution in the already-matching profile.

For reruns, do not restart based on an empty log poll. Observe the existing
sweep/session and PID state. Do not edit any arm's frozen sources while running.
`run.py --arm` refuses to overwrite an arm; `sweep.py --resume-from` requires
all earlier arms complete and their owned process trees stopped.

The previous cross-process layer20→42 drift remains unresolved. The pilot
proves the matching owner calls produce the same selection as full local work
within that forward; it does not establish global model determinism.
