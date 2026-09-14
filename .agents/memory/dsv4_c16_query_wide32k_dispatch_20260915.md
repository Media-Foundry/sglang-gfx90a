#32K follow-up: scheduler accounting and legacy MHC priority

Status:32K query-only ABBA restarted with corrected observation checks;
no completed32K throughput result yet. Existing16K acceptance remains
5657.4803 ->6321.9640 input tok/s (+11.7452 percent), with1M KV retained.

## First attempt stopped by an incorrect harness assumption

First service PID1398151 completed France (Paris) and one warmup4510.0877
input tok/s, then the driver required `prefill mix-reuse4 ... group=8` hits.
That assertion failed. No timed A/B legs or long quality waves completed.
The driver's finally block stopped the service; no owned workers remained
and all8 GPUs were verified free. This was not an OOM or a GPU computation
exception. Shutdown's child-exit messages must not be mistaken for the cause.

The first attempt, run script and logs were preserved by explicit renames,
not deleted. Failed archive19 files,1,228,298 bytes, SHA256
`9dae50c0874c02bea1b36cfb051c7bee3be19b0719b6f28f1c72214adc40f9a8`.
State absolute paths describe their original location before renaming.

## Source explanation: request count is not token-row count

Under32K budget, each~32K request consumes an entire admission. Model MHC
calls pass `global_batch_size=forward_batch.batch_size`. Thus the one-request
forward passes1 even though it processes~32768 token rows.

`mhc.py::_mhc_fusion_admitted(1)` unconditionally admits the legacy single-
request path. `mhc_fused_post_pre` calls the fused split-K tail before
`gfx90a_mhc_pre_mix_from_partials_triton`; a successful tail returns early,
so configured pre-mix8 is never consulted. The fallback split-K pre-mix can
also preempt reuse.16K admissions had2 requests and missed this old branch.

Launcher defaults enable SPLITK_MHC_PRE_MIX and FUSED_MHC_SPLITK_TAIL, and
FP16_MHC_DOT defaults1. The fused tail selects `fn_fp16` when that option is
on. This is an existing path, not a new quantization introduced by query
reuse. The first failed process did not log the actual dtype; the retry now
records path/M/request batch/dtype on every rank to verify it directly.
Do not label a configured reuse8 flag as proof that its FP32 path ran.

## Observation-only correction

Added one-time, active-original-prefill-only logs to successful legacy
split-K return paths. `audit_mhc_logging.py` strips only the new helper/state
and its two calls and compares the complete mhc.py AST to39bc0e1383: exact.
No math, kernel arguments, tensor operations, return values or dispatch
choices were changed.27 CPU tests and56 subtests passed; one pre-existing
unknown asyncio_mode pytest-option warning remains.

The32K driver now requires actual per-rank `fused_tail`,batch1 hits and saves
`mhc-observed.json`. Its analyzer requires the same observed path/dtype across
A1/B/A2. Both arms preserve the existing MHC behavior; only wide C4 is toggled.
Changing large-M priority to use FP32 reuse8 is a separate future numerical
and speed oracle, not something to mix into this query-only ABBA. Do not
globally disable legacy flags and accidentally change C1 AR/DSpark.

## Separate accounting clarification

Scheduler `#new-token` comes from page-rounded prefill budget accounting.
It is not exact model M. Current32K log includes real projection M32767 while
the admission reports32768. CPU audit uses the current `ceil_paged_tokens`
method and verifies the call/accumulation/logging chain:

| Input manifest | Actual input tokens | Page-rounded sum |
|---|---:|---:|
| C16x16K |262141|262144|
| C16x32K |524286|524288|

Client rates always used actual input IDs and wall time, so existing rates
are unchanged. Equal log histograms do not prove equal actual M, membership
or row positions. The32K analyzer now uses `scheduler_admissions` and
`identical_scheduler_admission_counts`, explicitly leaving actual-M equality
unproven.16K's preserved historical archive has an over-strong field name;
its memory report now documents the narrower interpretation.

Real mixed-prefix plan (not yet executed) preserves full input IDs and primes
page-aligned0/25/50/75-percent prefixes. If all planned prefixes hit, it leaves
328190 new tokens:10x32768 plus510. Actual cache counts may differ and must be
recorded; do not infer them from the intended percentages. Current native
TP8 raw grouped runtime-M covers129..1023, but actual larger tails and the
indexer/MHC selectors still require execution evidence.
