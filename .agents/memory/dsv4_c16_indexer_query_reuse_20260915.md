# Original V4 TP8 prefill: four-query indexer K reuse (2026-09-15)

## Status and objective

Positive component result, integrated candidate is **default-off**. Full C16
service ABBA has started but is not yet accepted. The production E2E checkpoint
remains **6477.19 input tok/s** from the preceding FP32 pre-mix experiment.
Do not multiply that speed by the component speedup below.

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
`SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4=1`. No launcher default has changed.
Eligibility requires existing original-V4 ordinary EXTEND/role/unified-pool
guards, canonical SGL Top512, enabled trivial and empty-tile skips, TP8 and
attention TP8, EP1/CP1/PP1, M8192..65536, no TBO/rewritten batch or graph capture.
The wrapper initially admits C4 width512..2048 and BLOCK_S16. This is coverage
for the measured C16 x8K task, not a claim for arbitrary long-context shapes.
Decode, DSpark target/draft, V4.1 and TP4 remain outside the selector.

CPU selector/wrapper and MHC scope tests: 17 passed, 23 subtests, one unrelated
pytest asyncio_mode configuration warning. Actual eight-rank hit logs will be
required in service; flags alone are not evidence of a fast-path hit.

## Ongoing E2E protocol

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
