# TP4 concurrent decode: new C4 entry divergence

Continuation from6c459b84b3. Previous turn was progress: prefill ordering cause
isolated, fixed-order opt-in path verified for C1 and heterogeneous first logits.
Goal still active: C16 multi-token drift remains; no TP8 migration.

## Fixed batch experiment

`check_dsv4_tp4_m32_next_token.py` now accepts `--tokens` (default1).
One batched HTTP payload preserves input order; fresh salt per request.
16 distinct2304-token code requests,32 outputs,3 rounds:
`/tmp/dsv4_tp4_fixed_batch_{1,2,3}.json`.
Rounds1/2 match only8/16 complete output sequences,0/16 full logprob sequences.
Thus variable network arrival alone does not explain residual drift.
First changed returned logprobs versus round1 occur at output indices4–7;
most at4 (zero-based), the next C4 update after prefill.

## First divergence at absolute2307

Service `/tmp/dsv4_tp4_decode2307.log`: same nativeTP4EP1/MFMA32+64/
canonical-order1/overlap-off profile, with bounded debug snapshots of M16
and position2307 only. Two fixed input batches,8 output tokens.
`/tmp/dsv4_tp4_decode2307_comparison.json` compares646 stage snapshots.
All captured stages before layer8, and layer8 Q, are exact.
First difference: layer8 attention core, max abs0.125, relative L2~0.01646.
This is not merely final greedy amplification of a one-ULP output change.

## Indexer capture disproves cutoff-tie hypothesis for this sample

`/tmp/dsv4_tp4_decode_ties/replay_{0,1}.pt`, layer8/M16/position2307.
Logits buffer shape `[16,16384]`, valid sequence length577 per row.
Always mask invalid logits columns before comparison (unused capacity is not
an output correctness contract).

- The first576 valid scores are exact for all16 requests.
- ONLY column576 (the newly compressed577th key) differs for every request.
- Valid-score max abs difference2.0498.
- Page-table inversion proves logical selected set changes for request9.
- No valid row has a score tie at the512/513 cutoff in this capture.

A separate all-equal-score M16/L577 synthetic test does expose nondeterministic
tie selection in the atomicTopK kernel (5/16 rows differ despite order1).
A stable-score-sort oracle fixes that synthetic case, but was removed from the
working topk wrapper because it does NOT explain the real new-key drift above.
Do not conflate the two causes.

## Next probe

Local default-off `SGLANG_DSV4_DEBUG_COMPRESS_REPLAY_DIR` captures layer8,
M16,position2307, first two visits, both core and indexer compressor:
x, projected kv_score, paired current/previous state pages, APE, normalized
compressed result; indexer store also captures destination and raw cache page.
For page comparison, only written key/scale regions are meaningful; a new page
contains unwritten bytes. Service log `/tmp/dsv4_tp4_compress2307.log`.

That probe produced no snapshots: the active model uses CompressorV2's HIP
override, not the older forward_native route instrumented there. The unused
compressor diagnostics have been removed; do NOT infer compressor equality
from their absence. The indexer capture diagnostics were also removed after
preserving their evidence in /tmp.

## Concrete reader/writer layout mismatch found

`triton_fused_store_indexer` selects `PRESHUFFLE_TILE=16` when
`aiter_can_use_preshuffle_paged_mqa()` is true. Its key bytes are ordered
`[token_tile,column_tile,token_in_tile,column_in_tile]`; FP32 scales remain
at page-byte8192 plus4*slot. However `_fp8_paged_mqa_logits_kernel` always
loaded key bytes as `slot*128+column`. The torch fallback also interpreted
the preshuffled bytes as contiguous rows.

On a newly allocated page with only slot0 written, the old linear reader
consumes slot1–7's unwritten data instead of all128 dimensions of slot0.
This explains why old full-page scores can repeat while the new577th entry
depends on allocator history and other requests.

`check_dsv4_indexer_cache_layout.py` independently reproduces the defect:
same valid K/Q/weights, one valid token per request, sixteen requests; change
only unwritten cache bytes from zero to random finite FP8 bytes.

- Before fix:16/16 valid logits change, max abs110.0656357.
- After layout-aware address calculation: exact outputs, max abs0.
- Added the inverse tile permutation to the torch fallback too.
- Extended to65-token/two-page inputs and a FP32 oracle over actual stored
  quantized K/scales. Using x/scale as an independent quantization oracle is
  inappropriate here: the writer uses x*reciprocal(scale), which can round
  differently at an FP8 midpoint. Reader reference must hold stored bytes fixed.

E2E under test: `/tmp/dsv4_tp4_layout_fix.log`, no dump flags, same TP4
profile and canonical-order1. No default promotion or TP8 migration yet.

## Verified after layout fix

- Partial-page invariance and FP32 reader reference pass at65 tokens (two
  pages), both preshuffled and legacy linear layouts.
- Torch fallback (`FULL=0`, --fallback) also passes65-token unwritten-byte
  invariance; no reader path is left assuming linear layout under preshuffle.
- `/tmp/dsv4_tp4_layout_fix_{1,2,3}.json`: fixed HTTP batch of16 diverse
  requests,32 outputs each.15/16 requests have exact full logprob sequences
  across all three rounds. Only request index2 differs, starting at output0
  (prefill), not universally at the new C4 entry anymore.
- `/tmp/dsv4_tp4_layout_fix_request2_{1..5}.json`: request2 isolated at C1,
  fresh salts,32 outputs. All five output-ID sequences match, but first
  logprob varies: -0.03036445/-0.03024371/-0.04374360/-0.04362206/-0.04362206.
  Thus the final residual is not specific to concurrency. Trace this prompt's
  first prefill divergence; the already-proven tie-selection issue is a
  candidate, not yet established as its cause.

The layout correction is unconditional for matching writer layouts; it does
not change quantization precision, selected attention semantics, or weights.
The exploratory stable-score-sort mode2 was not retained or used for E2E.
