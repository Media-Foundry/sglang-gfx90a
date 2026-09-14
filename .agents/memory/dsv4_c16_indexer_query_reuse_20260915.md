# Original V4 TP8 prefill: four-query indexer K reuse (2026-09-15)

## Status and objective

**Accepted after complete v2 ABBA: 6480.2232 -> 6773.2707 input tok/s,
+4.5222%.** Default-enabled only in the combined TP8 multi-request and
prefill-throughput launcher profile, preserving explicit0 and the conservative
runtime guards. Do not multiply E2E speed by the component speedup below.

The previous goal turn was progress: exact FP32 pre-mix integration, accepted
+9.09% C16 ABBA, profile default, evidence and fork/main push. This turn tests
the reviewer's still-untried multi-query K reuse rather than repeating the
already accepted prefill empty-tile optimization.

## Kernel contract

One CTA considers four consecutive query rows for a BLOCK_S16 K tile. Active
rows must have length >512 and reach that tile. If their physical page IDs
agree, load/decode K and scales once, then compute **four separate original
64-head dot/reduction operations**. If IDs differ, each row loads its own page.
Inactive/trivial rows keep defined zero scratch. Each row preserves its own
length, Q, weights, output scores and subsequent logical/physical Top-K.
No shared representative selection, KV truncation or compressor change.

The page test is tile-local; rows may belong to different requests and still
share a real physical prefix page, while unrelated pages fall back correctly.
BLOCK_S16 stays within a 64-token page. The runtime wrapper accepts strided
page-table rows (unit inner stride), and does not copy/repack Q or cache.
Unsupported contracts return None and use the original implementation.

No intermediate [query,key,head] global tensor: each 64-head reduction occurs
inside the CTA and only existing [query,key] scores are written. Same output
scratch capacity; no larger resident KV/weight workspace.

## Component evidence (physical GCD4 only)

Synthetic 8K causal requests reuse the existing empty-tile fixture with
independent per-request pages. Both arms include deterministic logical AND
physical Top-K. Crucially the control is the current nonempty implementation,
not the old rectangular kernel. Query projections and compressor are outside
these timings. The fixture is not a real model Q/K capture.

| Candidate | M | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|---:|
| 2-query screen |8192|7.95405|6.10388|1.3031x|
| 4-query standalone full |8192|7.94289|4.97090|1.5979x|
| 4-query standalone full |32768|31.72237|19.92051|1.5924x|
| Integrated, final strided fixture |8192|7.94639|4.99626|1.5905x|
| Integrated, final strided fixture |32768|31.70947|19.93185|1.5909x|

Three ABBA cycles per component, five graph iterations per observation. The
four-query standalone reports 65 registers, zero spills and 8192 bytes LDS;
that compiler metadata is not a separately measured full-service occupancy.

The final integrated test passes:

- 100 mutations at M8192 and 100 at M32768, varying Q/K/head weights, causal
  lengths and physical layouts: all FP32 score bits and both Top-K arrays exact.
- 100 fixed graph replays at each large shape, plus boundary/cutoff-tie cases.
- 100 additional mutations with M17, width577, partial last tile, unrelated
  per-row pages, repeated/shared pages and zero-Q ties. Page-table stride is
  [13,1] although logical width is10; all scores and Top-K outputs exact.

Current successful tests use torch.float8_e4m3fn, BF16 dot and preshuffle16.
The first preflight incorrectly assumed FNUZ and asserted before kernel launch;
its log is retained (`bq2.log`). The test and kernel now derive FP8 interpretation
and preshuffle from the same runtime helpers as the current control. Do not
report that failed preflight as a GPU correctness test.

The first integrated run (`runtime.json`) tested contiguous ragged tables; the
final run (`runtime-strided.json`) adds the stride regression. All are retained.

## Integration and isolation

New module `gfx90a_indexer_query_reuse.py`; flag
`SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4=1`. Initially default-off; after the completed
v2 ABBA below, the matched launcher profile opts in. Other profiles stay off.
Eligibility requires existing original-V4 ordinary EXTEND/role/unified-pool
guards, canonical SGL Top512, enabled trivial and empty-tile skips, TP8 and
attention TP8, EP1/CP1/PP1, M8192..65536, no TBO/rewritten batch or graph capture.
The wrapper initially admits C4 width512..2048 and BLOCK_S16. This is coverage
for the measured C16 x8K task, not a claim for arbitrary long-context shapes.
Decode, DSpark target/draft, V4.1 and TP4 remain outside the selector.

CPU selector/wrapper and MHC scope tests: 17 passed, 23 subtests, one unrelated
pytest asyncio_mode configuration warning. Actual eight-rank hit logs will be
required in service; flags alone are not evidence of a fast-path hit.

## E2E protocol

Directory `.agents/experiments/dsv4_c16_indexer_qreuse_20260915/`.
Three fresh owned services A1 -> B1/B2 -> A2, three timed waves per leg,
excluded warmup per process. Both arms keep all previous accepted MHC and
empty-tile optimizations. Only the new query-reuse flag differs.

Same 16 real code requests, 131069 explicit input IDs, no prefix hits,
max_new_tokens1, chunk32768, native TP8/EP1/no-A2A and actual 1M logical KV.
Each process separately runs France plus two C16 x128-output quality waves
with input echoes and completion-ID/text checks. Analyzer checks immutable
runtime hashes, equal timed shape counts and all quality-wave comparisons.

Whole-model drift is still open. The preceding MHC experiment had control
case8 and candidate case4 wording changes, not an input-ID mismatch. This
candidate's local exactness must not be misreported as global determinism.

## First service attempt: rejected harness error, not a GPU failure

A1 completed at 6482.758 input tok/s, reproduced the preceding checkpoint,
and repeated all16 quality outputs exactly. B completed only its excluded
warmup (5426.653 input tok/s); it printed eight query-reuse hits without rank
prefixes. The harness incorrectly demanded `TP0]` .. `TP7]` in those print
lines and raised AssertionError immediately after warmup. No formal B wave ran.
The owned shutdown completed with no remaining children; AMD-SMI reported all
eight GCDs empty. All first-attempt files remain in the original directory.

Fixed the actual one-time hit message to include `get_parallel().tp_rank`.
A CPU regression evaluates that source print expression for all eight ranks
and checks the same strings used by the service verifier. Full CPU suite now
passes21 tests and23 subtests (one unrelated asyncio_mode warning).

Since indexer.py's source hash changes (logging only), do NOT splice the old A1
into a supposedly immutable-source ABBA. A completely new A1/B1/B2/A2 attempt
completed under `.agents/experiments/dsv4_c16_indexer_qreuse_v2_20260915/`.
The kernel/arithmetic source is unchanged.

Read-only cross-cycle input audit (`compare_prior_quality.py`) checked the14
available completed quality waves before the second attempt. Explicit input
IDs agree throughout. Two output variants occur for case8 across controls and
candidates. Case4's second variant occurs only in premix/B/1 in that snapshot;
the audit does not establish its cause. These are cross-cycle observations,
not a controlled numerical attribution or a new service performance result.

## Completed v2 ABBA and default decision

| Leg | Input tok/s per timed wave | Median |
|---|---|---:|
| A1 |6485.486 /6478.870 /6481.536|6481.536|
| B1 |6772.645 /6776.278 /6775.050|6775.050|
| B2 |6771.492 /6770.280 /6771.902|6771.492|
| A2 |6478.910 /6477.334 /6481.115|6478.910|

Mean leg medians: **6480.2232 -> 6773.2707 input tok/s (+4.5222%)**.
Mean of leg median-request TTFT: **12.73672 -> 12.18758 s (-4.3115%)**.
Whole-wave median durations A1/A2 20.22190/20.23010 s, B1/B2
19.34584/19.35600 s; do not confuse wave time and median-request TTFT.

All four legs log twelve four-request/M32768 forwards (four per wave); actual
query hit logs include M32766. All eight B ranks are identified explicitly.
Every arm retains actual post-fused4 and mix-reuse4 hits. Runtime capacity is
1,048,576 logical KV, no prefix hits, and the source/input manifests match
across all three fresh services. Warmups are excluded. The first failed attempt
is not used in these medians.

All96 input echoes match their explicit IDs, all completion counts are128, and
completion IDs decode to response text. All three France sentinels answer Paris.
Each pair of **first** quality waves (A1/B/A2) matches16/16. B repeats16/16;
A2 repeats16/16; A1 repeats15/16. Its sole variant is case5 after57 common
tokens (IDs18586/666), a differently structured explanation of sparse-attention
metadata. Both B waves match both A2 waves16/16; differences against A1's second
wave are confined to that same case5. All16 candidate first-wave excerpts and
the differing control excerpt were inspected: coherent, task-related analysis,
no obvious repetition collapse. This limited128-token check is not full code
correctness, and the optimization does not prove whole-model determinism.

The default-line change happened **after** immutable-source ABBA, which used
explicit0/1. It changes profile selection only; no arithmetic source changed.
There is no separate post-default fresh-service performance claim. The kernel
keeps the original full score scratch size and all native per-row selection
semantics. No benefit is claimed for decode, speculative execution or C4
widths outside the admitted512..2048 range.

## Final validation and evidence

Post-default CPU checks: **22 passed, 31 subtests**, one unrelated pytest
asyncio_mode configuration warning. Profile tests preserve explicit0/1 and
require both profiles, TP8/EP1/no-A2A. `bash -n` and `git diff --check` pass.

The v2 evidence archive contains97 files, 2,377,564 bytes, SHA256
`3a2f2d1f8971b8149c343e2e561eb4070c523a82fffecfec5ccdb2d779385f8b`.
It packages all accepted raw waves, quality responses, source/input manifests,
actual rank-hit logs and clean shutdown records, component JSON/logs, and the
rejected first-attempt harness logs separately. No model tensors. All three
accepted services exited with no remaining owned processes; AMD-SMI confirmed
no processes on all eight GCDs after the run.

This is another measured reduction in avoidable prefill work, not proof of a
hardware limit. Remaining candidates must be evaluated against this complete
6773 checkpoint. A small BQ8 component screen may test further tile reuse while
keeping separate per-row dot/reduction; a revised critical-path profile should
precede broad MoE/indexer ownership work. The old 2.64 s logits budget predates
this reuse win and must not be carried forward unchanged.
