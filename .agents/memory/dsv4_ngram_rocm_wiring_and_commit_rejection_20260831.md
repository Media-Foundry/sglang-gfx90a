# DSV4 NGRAM ROCm wiring and commit-path rejection (2026-08-31)

## Scope

- Hardware for the service experiment: physical GCDs `4,5,6,7` via
  `HIP_VISIBLE_DEVICES=4,5,6,7`.
- Model: original DeepSeek-V4-Flash checkpoint, TP4/EP1.
- Proposal experiment: NGRAM, three draft tokens, breadth forced to one.

## ROCm AOT defect fixed independently

The ROCm `common_ops` source list and registration table omitted
`csrc/speculative/ngram_utils.cu` and
`reconstruct_indices_from_tree_mask`, although the Python NGRAM worker calls
that operator. CUDA already included and registered it. After adding the ROCm
source and registration, the physical-GPU-4 test
`python/sglang/kernels/aot/tests/speculative/test_ngram_utils.py` passed.

This is a generic ROCm NGRAM build/wiring fix. It does **not** make NGRAM safe
for DeepSeek V4.

## DSV4 experiment result

The service loaded the model and captured target-verify graphs successfully,
but the first real France request failed before returning a token:

```text
AttributeError: 'DeepSeekV4TokenToKVPool' object has no attribute 'move_kv_cache'
```

The failure is structural rather than a missing one-line adapter:

- Generic NGRAM verifies a candidate tree and then relocates the accepted KV
  slots with `move_accept_tokens_to_target_kvcache`.
- The DSV4 HIP target-verify backend currently constructs linear causal
  metadata and does not consume the NGRAM tree mask. General BFS siblings
  therefore cannot be verified correctly.
- DSV4 compressor state is indexed by request and logical position. Siblings at
  the same depth can overwrite the same c4/c128 state ring slot before the
  post-verify move occurs.
- Even the forced breadth-one chain needs boundary-aware moves for c4 attention
  KV, c4 indexer payload+scale, and c128 attention KV, plus rejected-c128-state
  cleanup. A plain `loc // ratio` copy is wrong for non-boundary tokens and for
  overlapping source/target ranges.

Consequently the temporary DSV4 NGRAM launch/whitelist changes were removed.
No NGRAM throughput number is valid.

## Required correctness work before reconsidering

1. Fail loudly for DSV4 tree NGRAM (`max_bfs_breadth > 1`) until target verify
   consumes tree masks and compressor state becomes branch-local.
2. For a possible breadth-one profile, implement staged, boundary-masked moves
   across 3/4/5 and 127/128/129, including packed indexer scales.
3. Clear rejected c128 draft states, matching the EAGLE-family cleanup contract.
4. Compare cached NGRAM logits against native cached AR and full recompute at
   absolute positions 3/4/5, 127/128/129, and 511/512/513.
5. Require eager/graph parity, BS1/4 batch-filter tests, France first-nine exact,
   and real heterogeneous code-request E2E before any performance claim.

## Operational load balancing

After rejecting NGRAM, the validated DSpark gamma-1 TP4 service was restarted on
physical GCDs `4,5,6,7`. France first-nine and semantic Paris checks passed.
Three rounds of 32 distinct code prompts produced:

- aggregate tok/s: `691.76 / 690.47 / 678.59`, median `690.47`;
- resident-window tok/s: `786.87 / 788.03 / 763.61`, median `786.87`;
- scheduler tok/s: `777.62 / 765.99 / 746.27`;
- mean accepted length: `1.763 / 1.769 / 1.759`.

All 96 requests completed 256 tokens with `finish=length`. Cross-round greedy
hashes are not bitwise stable under the existing asynchronous scheduling order;
the separately run France oracle remained exact, matching the project's current
E2E acceptance rule.
