# DeepSeek V4 C4 indexer graph-width cap

Date: 2026-08-30

## Finding

`PagedIndexerMetadata.max_c4_seq_len` used the model-sized page-table width.
For the 1,048,576-token model this made every captured sparse decode graph use
roughly 262,144 C4 logits columns, even though the serving profile had a
32,768-token KV pool and therefore could never make more than 8,192 C4 rows
live. CUDA/HIP graph replay preserves the captured shape; masking the inactive
tail did not shrink the logits launch or the following Top-K workspace.

This is separate from the raw-vs-C4 threshold bug fixed in `9cee417f3e`:

- the sparse selector must compare raw sequence length against
  `index_topk * compression_ratio` (512 * 4 = 2048 raw tokens);
- once sparse is selected, its fixed graph width should not exceed the maximum
  C4 rows physically representable by the complete KV token pool.

## Fix

Added `SGLANG_DSV4_INDEXER_MAX_C4_SEQ_LEN`. The ROCm DSV4 launch harness sets
it, unless explicitly overridden, to:

```text
ceil(MAX_TOTAL_TOKENS / 4)
```

`PagedIndexerMetadata.max_c4_seq_len` takes the minimum of this value and the
model/page-table capacity. This does not truncate a valid request: no single
request can hold more tokens than the whole KV pool. Raising
`MAX_TOTAL_TOKENS` automatically raises the graph bound.

## Sparse-only isolation A/B

Profile: TP4/EP1, original checkpoint precision, native AR, 4 gfx90a GCDs,
M32 graph tiers `1,8,16,24,32`, 32 real heterogeneous prompts, 128 generated
tokens, 32,768-token KV pool. The full sparse graph was deliberately forced at
short context to isolate its fixed cost.

Before the graph-width cap (after BF16 indexer projection caching):

```text
resident M32: 163.75 tok/s
```

With the 8,192-C4-row cap, three rounds:

```text
629.48 / 630.17 / 629.73 tok/s
median: 629.73 tok/s
speedup: about 3.84x
```

All 32 requests completed at length, the France sentinel passed, and the
32-request next-token teacher output matched the accepted reference.

## Formal dual-graph result

With the normal dense/sparse dual graph and the raw 2048 switch threshold:

```text
32 heterogeneous requests x 544 generated tokens
resident M32: 699.32 tok/s
aggregate wall-time: 674.39 tok/s
France sentinel: pass
next-token teacher: 32/32 match
```

The one-time 480--512 generated-token transition bin was 530.27 tok/s, then
512--544 recovered immediately to 685.25 tok/s. The old failure remained near
150 tok/s after the transition. Therefore the persistent collapse was the
oversized sparse graph, not VRAM eviction. The remaining one-bin transition is
a separate cache/compressor boundary cost.

Artifacts:

- `/tmp/dsv4_tp4_bs32_teacher_sparse_cap8192.json`
- `/tmp/dsv4_tp4_bs32_sparse_cap8192_short.json`
- `/tmp/dsv4_tp4_bs32_teacher_dual_cap8192.json`
- `/tmp/dsv4_tp4_bs32_dual_cap8192_544.json`

