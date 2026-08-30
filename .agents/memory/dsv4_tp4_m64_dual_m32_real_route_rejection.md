# TP4 M64 -> concurrent 2xM32 real-route oracle: rejected

Date: 2026-08-30

## Question

Could the accepted TP4/EP1 M64 routed stage be split into two independent M32
pipelines and overlapped on two HIP streams?  Earlier occupancy-only recorder
data was insufficient for this experiment because it does not preserve which
six experts each token selected or the corresponding router weights.

## Real runtime snapshot

A temporary, eager-only hook was added only to
`python/sglang/srt/layers/moe/moe_runner/aiter.py`.  On rank 0 it identified
layers by the stable `w13.data_ptr()`, counted M64 calls independently per
layer, and saved learned layer 34 / pass 20 to:

```text
/tmp/dsv4_tp4_m64_real_route.pt
SHA256 362c27bf3cbeb7d9618f1587b28426f33dc28d6627be0679a6a8c5b20289b884
```

The input was the checked-in 64-request heterogeneous token-ID manifest.  The
snapshot contains real runtime tensors:

```text
hidden_states  [64,4096] bf16
topk_ids       [64,6]    int32
topk_weights   [64,6]    fp32
unique experts = 166
```

The eager capture service completed 64/64 requests at 64 output tokens,
reported `finish=length` for every request, and passed the France first-nine-ID
sentinel.  The hook was then removed, the service stopped, and
`git diff --exit-code -- python/sglang/srt/layers/moe/moe_runner/aiter.py`
passed.  No snapshot instrumentation remains in production.

## Oracle

The one-shot standalone oracle was intentionally removed after the formal run;
it was not retained as production or repository tooling.  The run used the
real hidden values, exact per-token expert IDs, and exact router
weights.  Packed FP4 weights and scale tensors use the checkpoint layouts but
synthetic values; this is sufficient for the scheduling/scan oracle because
all three arms share the same read-only tensors and the split preserves every
token's exact arithmetic.  It runs the accepted kernel geometries:

- M64: DPP A4/R2/W8 gate, logical-scale row-prefetch A4/W4 down;
- M32: DPP A4/R2/W8 gate, logical-scale row-prefetch A4/W8 down;
- unchanged group-32 quantization and fixed-slot FP32 reduction.

The three arms are:

- **A**: one complete M64 routed stage;
- **S**: two M32 stages serially;
- **B**: two M32 stages on independent streams with independent activation,
  quantization, partial, and output workspaces; packed weights are shared
  read-only.

The offline route-aware partition is exactly balanced at 32/32 tokens.  A
deterministic seeded swap search exchanges one token from each side, evaluates
`sum_e ceil(count_left[e]/4) + sum_e ceil(count_right[e]/4)`, and retains the
lowest-cost partition.  It is diagnostic only and performs no D2H
synchronization in a service path.

## A4 scan result

| Partition | M64 scans | split scans | inflation |
|---|---:|---:|---:|
| original rows 0--31 / 32--63 | 182 | 236 | +29.67% |
| route-aware balanced 32/32 | 182 | 200 | +9.89% |

Thus route-aware partitioning can meet the predeclared `<=15%` scan-inflation
gate.  The important result is that scan count is not the limiting issue after
balancing.

## Formal timings

GPU 0, seven-round interleaved A/S/B/B/S/A, 30 timed iterations per sample,
one-sample trim at each tail:

| Partition | A trimmed mean | S trimmed mean | B trimmed mean | B vs S |
|---|---:|---:|---:|---:|
| original | 744.107 us | 877.990 us | 866.787 us | -1.28% |
| route-aware balanced | 743.942 us | 815.523 us | 798.481 us | -2.09% |

The balanced concurrent arm is **7.33% slower** than the single-M64 arm.  It
misses both continuation gates decisively:

```text
B <= 620 us             observed 798.481 us
B at least 25% vs S     observed 2.09%
```

The two halves compete for the same CU/HBM resources; independent stream
submission exposes almost no useful overlap.  Reducing padding/scan inflation
does improve the split itself (`866.8 -> 798.5 us`) but cannot compensate for
duplicated M32 grids and poorer whole-device work decomposition.

## Correctness and graph safety

- 100 activation mutations: A, S, and B outputs bitwise exact for both
  partitions;
- 1000 HIP Graph replays: bitwise stable;
- each M32 arm was captured as its own HIP Graph on its owning stream, then
  both graphs were enqueued before joining.  A parent-only multistream capture
  was explicitly rejected because ROCm warned that the parent graph was empty.

## Decision

Reject M64 -> 2xM32 stream overlap and do not connect it to production.  Even a
high-quality route-aware hypergraph partition cannot make concurrent M32 grids
beat the current single-M64 routed stage.  Future work should preserve one M64
launch/work decomposition and reduce work inside it rather than splitting it
into competing full-device pipelines.
