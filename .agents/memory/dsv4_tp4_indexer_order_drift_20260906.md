# Residual TP4 drift: indexer output permutation

Previous turn was progress: commit719de619e1 corrected an independently
reproduced divergent wave shuffle. This investigation addresses residual drift.

## First divergence in real fixed-input model forwards

Native TP4/EP1/no-A2A, original weights, GCD4–7, chunk2304, MFMA32+64 on,
overlap schedule off, fresh cache salt each request. No TP8 migration yet.

1. `/tmp/dsv4_tp4_firstdiv`: two full-prefill executions, rank0, last8 rows
   at all43 layers. First sampled difference is layer2 attention output.
2. `/tmp/dsv4_tp4_firstdiv_full`: complete2304 rows, layers0–2, additionally
   Q, attention core, inverse RoPE, wo_a, wo_b before/after TP reduction.
   All captured layer0/1 intermediates and layer2 Q are exact.
   First difference is layer2 attention core: max abs0.015625,
   relative L2 0.000117897. The218 differing rows begin at zero-based2051.
3. Config compress_ratios starts `[0,0,4,128,...]`. Position2051 gives
   513 C4 entries, the first position actually selecting Top512.

## Direct indexer evidence

`/tmp/dsv4_tp4_indexer_firstdiv/replay_{0,1}.pt`:

- Logits `[2304,576]`: bitwise identical, finite max abs0.
- Sequence lengths: identical.
- Physical page tables differ (first request pages1–9; second10–18), expected
  for fresh caches. Comparing physical indices directly is invalid.
- Inverting the page-table transform with page_size64 proves **logical
  selected sets identical for every row**, but **logical order differs**,
  starting at position2051.
- The gfx90a AOT top-k implementation uses atomicAdd to append selected IDs;
  its output ordering is unspecified. Changed order changes attention reduction
  grouping even with identical selected KV membership and indexer scores.
- Independent random-score oracle on GPU0 (M256,L576) confirms129 rows
  change order across two calls while every selected set stays identical.

## Candidate, still under E2E validation

`SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER=1` asks HIP top-k for raw logical
IDs, sorts them descending, and permutes physical IDs by the same permutation.
Padding -1 stays at the tail. Never sort physical addresses as the canonical
key: cache placement is not logical sequence order.

Real captured logits replay on GPU0:20/20 exact ordered outputs and selected
sets equal the original output. This does not address arbitrary equal-score
ties at the selection cutoff; investigate separately if observed.

Service under test: `/tmp/dsv4_tp4_canonical_indexer.log`. No dump hooks are
enabled for this E2E run. Canonicalization is currently opt-in, not yet promoted.

## E2E evidence after canonicalization

- `/tmp/dsv4_tp4_canonical_c1_256.json`: five rounds, fresh salts, all cached
  tokens zero, all256 output IDs identical. SHA256 for every round:
  `e046e7dcc43e8cf8eebdc5621bc31246819af0533970b7b06ed6b5d598d38139`.
- `/tmp/dsv4_tp4_canonical_logits_{1,2}.json`:16 distinct2304-token code
  requests, all16 rows now exactly match including returned top5 logprobs.
  The prior noncanonical run had differing logprobs for all16 rows.
- `scripts/rocm/check_dsv4_topk_order_replay.py`:100/100 synthetic replays
  exact with lengths spanning short/padded and selectedTop512 rows; logical
  selected membership preserved. Captured real first-divergence logits had
  no ties at the512/513 cutoff.
- `/tmp/dsv4_tp4_canonical_c16_32.json`: concurrent16 ×32-token ×3 rounds.
  First tokens exact, but full completions NOT exact. Differing request indices
  versus round0: round1 `[2,3,8,15]`, round2 `[2,8,11,14]` (zero-based).
  Input throughput2431.39/2465.59/2448.37tok/s; not an ABBA speed claim.
  Thus single-request full output and heterogeneous first-step logits recover,
  but concurrent decode still requires investigation. Do not promote this
  result to full TP4 correctness or migrate TP8 yet.

Do not use probe timings for performance claims. The current canonicalization
uses PyTorch sort/gather/copy; a fused HIP implementation can be optimized only
after maintaining the same logical-order contract.

## Working tree diagnostics

Bounded DEBUG_STAGE sampling/replay controls and DEBUG_INDEXER_REPLAY_DIR
instrumentation remain local diagnostic edits; they are OFF in the canonical
E2E service. The model file also contains unrelated pre-existing CK replay
hooks: preserve those. `compare_dsv4_stage_replays.py` compares saved tensors
with max-abs/relative-L2, not floating-point hashes.
