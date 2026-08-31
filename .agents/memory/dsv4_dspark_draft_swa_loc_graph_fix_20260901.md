# DSpark draft SWA-location CUDA Graph replay fix (2026-09-01)

## Scope

- Model: DeepSeek-V4-Flash, original checkpoint weights.
- Hardware: physical GCDs 4,5,6,7; TP4 / EP1 / no A2A.
- Workload: 32 distinct code/chat requests, 256 generated tokens, native target
  verification with DSpark gamma 5 and a compact M128 verify graph.
- The change is draft-only: native AR and target-model cached metadata retain
  their previous path.

## Root cause

`UnifiedKvMetadata.copy_` treats `swa_loc` as an assigned field. During CUDA
Graph replay, assigning a new Python tensor cannot update the tensor pointer
captured by the draft model's KV-store kernel. The gamma-5 draft graph could
therefore write through the synthetic capture-time SWA locations instead of
the live request slots and positions.

The fix makes DSpark draft `TARGET_VERIFY` recompute `swa_loc` from the
graph-stable live `req_pool_indices` and `positions` buffers inside the graph.
All other paths keep the existing cached fast path.

## Verification

The focused unit battery passed 10 tests:

```text
test_dspark_swa_loc_replay.py
test_dsv4_dspark_anchor_only_routed.py
test_dspark_ragged_extra_buckets.py
```

The new test proves that DSpark draft target verification ignores a stale
cached location and produces `[513, 898]` from live inputs, while native target
and non-target draft paths still reuse their cached tensor.

## End-to-end effect

With gamma 5, M128 compact verification, forced verify-budget fraction 0.6:

| configuration | resident BS32 tok/s | mean accepted length |
| --- | ---: | ---: |
| graph transformer, eager proposal, before fix | 782.09 | 2.553 |
| graph transformer, eager proposal, after fix | 862.44 | 3.047 |
| full folded graph, before fix | 706.09 | 2.316 |
| full folded graph, after fix (first round) | 1033.59 | 3.429 |

The full-folded same-path recovery is about 46%, and the isolated graph
transformer recovery is about 10%.

## Important remaining correctness limit

This fix does **not** validate the gamma-5 M128 anchor-only routed-MoE
approximation. A five-round run of that approximation had resident speeds
`1058.05 / 1008.67 / 1045.64 / 1105.01 / 1018.33 tok/s`, but only three rounds
answered Paris promptly; two entered a repeated France preamble. Disabling the
draft graph did not eliminate that variability.

Disabling anchor-only routed MoE restored strict France first-nine correctness
for all three rounds, but reduced the resident median to 678.60 tok/s and the
mean accepted length to about 2.61. Therefore:

- the `swa_loc` graph replay repair is accepted independently;
- gamma-5 anchor-only remains experimental and is not a correctness-approved
  service checkpoint;
- the stable production comparison remains the gamma-3 profile until a
  correct target-verification work decomposition is found.

Evidence files:

```text
/tmp/dsv4_gamma5_m128_graph_unfolded_swaloc_r1.json
/tmp/dsv4_gamma5_m128_fullgraph_swaloc_r1.json
/tmp/dsv4_gamma5_m128_fullgraph_swaloc_r5.json
/tmp/dsv4_gamma5_m128_eagerdraft_swaloc_r5.json
/tmp/dsv4_gamma5_m128_exact_eagerdraft_r3.json
```
