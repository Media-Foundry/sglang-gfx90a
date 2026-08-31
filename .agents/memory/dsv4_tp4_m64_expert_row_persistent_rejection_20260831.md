# TP4 DSpark M64 expert-row persistent A4 rejection (2026-08-31)

## Question

Could the accepted TP4 M64 A4 routed-FP4 kernels improve temporal weight-cache
reuse by mapping a wave/subgroup to `(active expert, row tile)` and scanning the
same expert's consecutive A4 chunks, without changing routed math?

This is deliberately different from the previously rejected expert-owner CTA
publication kernel.  It retains the gate G2080/W8 and down D832/W4 grids, does
not serialize a complete expert in one CTA, and uses no counters, epochs or
publication protocol.

## Oracle

Standalone-only files:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_row_persistent_oracle.cuh`
- `scripts/rocm/bench_dsv4_dspark_gamma1_m64_expert_row_persistent.py`

Both arms use TP4 `E256/M64/topk6/H4096/I512/N4096`, A4/R2 gate, logical W2
scales, group-32 INT8 intermediate quantization, FP32 `[M,T,N]` partials and the
same fixed-slot BF16 reduction.  The candidate preserves the gate DPP and down
subgroup16 reduction trees.  Metadata adds contiguous A4 block starts/counts
for every active expert.

All tests used only physical GPU 4 after `amd-smi process` showed no processes.
For each route, 100 randomized eager mutations and 1000 HIP Graph mutation
replays were bitwise exact for the BF16 intermediate, INT8 values, FP32 scales,
FP32 partial and final BF16 output.

## Results

Seven-round ABBA trimmed means, microseconds:

| route | active experts | A4 scans | stage | current | expert-row | change |
|---|---:|---:|---|---:|---:|---:|
| warm pass 64, layer 20 | 105 | 149 | gate | 388.313 | 541.356 | +39.4% |
| | | | down | 259.360 | 303.911 | +17.2% |
| | | | full | 664.813 | 861.865 | +29.6% latency |
| concentrated pass 27, layer 20 | 16 | 101 | gate | 314.401 | 448.939 | +42.8% |
| | | | down | 227.805 | 252.640 | +10.9% |
| | | | full | 558.985 | 714.517 | +27.8% latency |

The benchmark's throughput-style `gain_pct` was `-22.863%` and `-21.767%`
respectively.

## Decision

Reject expert-row persistence for the current A4 implementation.  Even the
most concentrated recorded route, with only 16 active experts and 101 A4
scans, regresses substantially.  Sequential chunk scanning does not recover
enough cache reuse to offset the lost independent row tasks and longer
per-wave dependency chain.  Do not add a production selector and do not retry
this mapping merely at M84/M96; a future attempt must change the kernel's
weight-load/accumulator decomposition rather than only task order.
