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

## Retry runtime witness

Retry PID1407476 started and answered France before32K prefill. At05:05:09,
all8 ranks logged `path=fused_tail rows=32767 batch=1 weight_dtype=torch.float16`.
Thus the legacy MHC priority and FP16 cached Fn selection are confirmed at
runtime, not just inferred from launcher defaults. Query-only A/B will preserve
this existing path in both arms; these32K measurements must explicitly state
that they do not use the FP32 pre-mix8 path of the16K multi-request forwards.
This does not change checkpoint files and is not a new precision reduction.

The finding motivates a separate large-prefill priority oracle: compare the
existing batch1 split-K path with FP32 reuse8 under the actual native-prefill
scope, while preserving C1 AR and speculative paths. Do not attribute all
same-configuration16K drift to this mechanism: those admissions use a different
request-count regime, and reduction/row-placement effects remain independently
documented. Full32K throughput and quality results are still pending.

## Retry A1 timing and prepared next oracle

Retry A1 warmup4515.0753, timed4608.3169/4608.1182/4608.9918 input tok/s;
median4608.3169. These are control-only figures, not an ABBA gain. First two
wave durations113.7695/113.7744s. Long quality checks are underway; no B/A2
result is claimed at this checkpoint. First32K control excerpts were manually
inspected for coherence/topic/repetition, not verified as factual code reviews.

Prepared `.agents/experiments/dsv4_prefill_mhc_priority_20260915/oracle.py`
for after this service sweep. It compares an allocating full MHC boundary:
legacy batch1/FP16 split-K versus large-prefill FP32 reuse8 priority, with
existing batch2 FP32 dispatch as reference. Captured residual/Fn/scale/base/
norm weights plus synthetic post/comb inputs are used, not fresh32K captures.
M1 must remain byte-exact; B must equal the FP32 reference and survive input
mutation/row permutation. No GPU execution or performance result yet.
Syntax/help and two CPU contract tests passed. The model-level sinkhorn arg
is20 (required by the gfx90a dispatcher), while both arms retain the existing
internal env override8. No production selector or numerical path was changed.

Retry A1 has now completed both quality waves and stopped cleanly.32/32 full
input echoes match;12/16 full128-token outputs repeat and15/16 first tokens
repeat. Changed cases/zero-based first positions:5/18,6/0,9/23,13/3. The
second-wave changed excerpts were also inspected and remain coherent code
reviews, not factual-accuracy or complete-answer proofs. This confirms that
single-request large-prefill admission does not eliminate same-configuration
drift; the MHC priority issue is not a complete explanation. B has started,
and the full32K ABBA is still pending.
