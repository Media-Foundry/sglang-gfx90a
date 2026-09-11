# DeepSeek-V4.1 storage audit: P2/P3 (2026-09-12)

## Scope and evidence

This audit uses the V4.1 checkpoint metadata in
`/media/PM983/deepseek-v4.1-flash/model.safetensors.index.json`, the current
on-disk loader/quantization code, and the clean-start logs under
`/tmp/sglang_v41/`.  It distinguishes live tensor storage from allocator
reserved bytes and from host mappings.  No claim below treats the startup
`mem usage` delta as a unique-storage census.

## P2: E8M0 scales are not globally expanded on the active AIter path

The checkpoint contains 23,563,015,184 E8M0 scale elements, or
21.944768 GiB at one byte per element:

| source | raw E8M0 bytes (global) |
| --- | ---: |
| routed FP4 expert scales | 16.215820 GiB |
| Engram embedding/table scales | 5.722502 GiB |
| attention and other small scales | 0.006446 GiB |
| **total** | **21.944768 GiB** |

For `SGLANG_USE_AITER=1` on HIP, `Fp8MoEMethod.create_fp8_moe_weight_`
allocates FP4 expert scales as `torch.float8_e8m0fnu`.  The AIter
`shuffle_scale_a16w4` transform preserves that dtype; it does not turn the
scales into FP32.  The current V4.1 service logs also show the AIter routed
layout being selected.

The two Engram tables are backed by one mmap byte buffer per table.  The
weight and E8M0 scale tensors are views into that same buffer, and
`EngramEmbedding._apply` removes them from the normal module device move.
The observed clean-start logs report, per TP rank:

```text
engram host table layer 1:  12085 MiB resident, 12084 MiB huge, pinned
engram host table layer 14: 12085 MiB resident, 12084 MiB huge, pinned
```

The exact table arithmetic is:

```text
layer 1: 384,006,168 rows × (256 FP8 weight + 8 E8M0 scale) = 94.415274 GiB global
layer14: 384,016,682 rows × (256 FP8 weight + 8 E8M0 scale) = 94.417859 GiB global
TP8 private shard total: ≈23.604142 GiB host per rank for both tables
```

Thus the host table is large, pinned RAM, but not a GPU tensor.  Private
layout shards rows across ranks; it does not duplicate physical table pages.

The often-quoted `~5.93 GiB/GCD` scale concern is real only for a fallback
that expands the **un-padded local MoE E8M0 scales** to FP32.  With local
intermediate 288, 40 routed layers, that one-to-four-byte increment is about
5.93 GiB/rank.  It is not present in the active AIter E8M0 allocation.
Dense non-expert scales are only about 6.45 MiB raw globally (about 25.8 MiB
if represented as FP32), so they cannot explain a multi-GiB discrepancy.

## P3: the large persistent cost is padding, not a hidden old-reference copy

V4.1 has 384 routed experts, hidden size 5120, and local intermediate 288 at
TP8.  The AIter path pads the local intermediate to 384 (`fp4_k_align=128`)
before creating its final weights/scales.  Per rank and per routed layer:

| representation | weight bytes | E8M0 scale bytes | total |
| --- | ---: | ---: | ---: |
| local 288 (no AIter pad) | 0.791016 GiB | 0.049438 GiB | 0.840454 GiB |
| AIter 384 padded | 1.054688 GiB | 0.065918 GiB | 1.120605 GiB |

Across 40 layers this is approximately:

```text
current padded AIter expert storage: 44.824218 GiB/GCD
un-padded FP8-scale equivalent:      33.618164 GiB/GCD
padding overhead:                    11.206055 GiB/GCD
```

The table and totals above are the nominal 384-wide layout.  The installed
machine uses the legacy AIter `shuffle_scale_a16w4` entry point, which requires
the W2 group axis to be padded from 12 to 16 by the compatibility shim.  Thus
the *active* post-shuffle scale total is 2.929688 GiB/GCD (rather than the
nominal 2.636719 GiB/GCD), and active routed expert storage is about 45.117188
GiB/GCD.  This extra 0.292969 GiB/GCD is a concrete scale-layout overhead, not
an FP32 expansion.

That overhead is a real live-parameter cost required by the current AIter
kernel geometry.  It is distinct from the ~5.93 GiB fallback FP32-scale
increment.  A non-AIter local-288 path with FP32 scales would be about
39.550781 GiB/GCD for routed expert tensors, before the rest of the model.

During `process_weights_after_loading_block_quant`, padding and shuffle calls
temporarily create replacement tensors.  The old tensor is released when the
parameter is rebound and the function returns; `view(float4...)` does not copy
storage.  `w2_weight_scale_logical` is an explicit extra clone, but it is only
created under the direct-debug scale-cache flags and is not enabled by the
V4.1 AIter launch.  For the V4.1 padded shape it would be roughly 0.88
GiB/GCD over 40 layers, so it must stay off unless a direct kernel needs it.

The allocator can retain released replacement blocks in `reserved` memory,
but SGLang's `get_available_gpu_memory()` calls the device `empty_cache()`
before reporting the post-load delta.  The failed HBM-Engram constructor also
reported only 73.71 MiB reserved-but-unallocated versus 62.71 GiB allocated,
which argues against a multi-GiB fragmentation-only explanation for that OOM.
The definitive live/reserved breakdown is supplied below by a successful
post-load census (`memory_allocated`, `memory_reserved`, and unique
`untyped_storage().data_ptr` bytes).

## Startup note

One earlier test used the invalid CLI spelling
`--max-total-num-tokens 1024` and failed argument parsing.  A separate
1024-token attempt using the valid `--max-total-tokens` spelling loaded weights
but rounded the SWA pool to zero, then correctly rejected it because the
admission floor is `sliding_window_size (128) + page_size (256)`.  These are
pool-configuration failures, not evidence of a P2/P3 storage failure.  The
successful census below used the valid `--max-total-tokens 8192` setting.

## Runtime unique-storage census (successful 8192-token startup)

On 2026-09-12, a clean TP8 start with private host tables and
`SGLANG_DEBUG_MODEL_STORAGE_AUDIT=1` logged the following on every rank (the
minor rank-to-rank allocator differences were below the displayed precision):

```text
CPU tensors:   logical=23.604 GiB, unique=23.604 GiB, 2 storages / 4 views
GPU tensors:   logical=68.208 GiB, unique=53.958 GiB, 1610 storages / 1724 views
GPU allocator: allocated=54.156 GiB, reserved=62.211 GiB, peak=56.117 GiB
routed weights=42.188 GiB, routed E8M0 scales=2.930 GiB
direct W2 scale clones=0, GPU Engram=0
host mmap=23.604 GiB across 2 mappings
```

The source log is `/tmp/sglang_v41/storage_audit_retry2_20260912.log`.
The 14.25 GiB difference between logical GPU bytes and unique storage is
mostly registered aliases/views (8 shared storage groups), not another copy.
Only about 0.198 GiB separates unique registered GPU storage from allocator
`allocated`; the remaining ~8.055 GiB in `reserved` is allocator cache/free
space.  The model-storage walk runs before KV-pool allocation, so these values
must not be conflated with the later static pool reservation (~60.6 GiB seen
by `amd-smi`).

## Separate pool-sizing coverage gap (not a duplicate-weight finding)

The current `DSV4PoolConfigurator._get_bytes_per_full_token()` accounts for
SWA/C4/C128 pools, but its current on-disk implementation does not add the
V4.1 ratio-1/2 source pools.  `DeepSeekV4TokenToKVPool` does allocate those
additional pools, using `full_size`, and also allocates their FP4 index pools.
For this checkpoint the four source layers are `[2, 8, 14, 20]` with ratios
`[2, 2, 2, 1]`; their omitted estimate is `(584+68)/2 * 3 + (584+68) =
1630 bytes/full-token` before page padding.  The current log reports only
`2336 bytes/full-token` (the 10%-SWA term), so an unconstrained auto-sized
pool can overestimate capacity by roughly 70%.  At `full_size=498688`, the
omitted low-ratio KV+index buffers are about `0.759 GiB/GCD` after page
rounding.  The successful audit used an explicit `--max-total-tokens 8192`,
where this is only about `0.013 GiB/GCD`, so it did not exercise the dangerous
large-capacity case.  This is a sizing/accounting issue to fix and test
separately; it is not evidence that model weights are duplicated.

## Decision

1. Do not add a scale-expansion workaround: active AIter scales are already
   E8M0 and Engram tables are host-backed.
2. Treat AIter `288→384` padding as the first real memory optimization target;
   removing it requires a kernel/layout contract, not allocator cleanup.
3. Keep direct logical-scale clones disabled for V4.1.
4. The successful 8192-token startup already recorded a gated
   unique-storage/allocated/reserved audit; use it as the baseline for any
   future padding or offload change.
