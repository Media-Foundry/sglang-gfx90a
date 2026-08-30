# TP4 M32 attention--MoE cross-layer wavefront: rejected

Date: 2026-08-30

## Question

Could the accepted TP4/EP1 M64 native-AR decode graph be represented as two
independent M32 microbatches, then pipeline microbatch A's attention at layer
`L+1` against microbatch B's routed/shared MoE at layer `L`?  At 64 requests
and 43 layers, 1500 token/s requires a two-microbatch layer pair no slower than
about 0.992 ms.

This is not equivalent to the rejected concurrent `2 x M32` routed oracle
(MoE versus MoE), the gate/down producer-consumer pipeline, PP2, or ordinary
SBO.  However, an older real TP8/M32 diagnostic had already found attention
`1.915 ms`, MoE `1.541 ms`, serial `3.253 ms`, and overlap `3.396 ms`.  The
current experiment tests whether the substantially newer TP4 kernels change
that resource-contention result.

## Optimistic single-GCD oracle

The production-disconnected benchmark is:

```text
scripts/rocm/bench_dsv4_tp4_m32_attn_moe_overlap_oracle.py
```

Inputs and kernels:

- real layer-20 M32 hidden/Q/attention tensors and BF16 projection weights from
  `/tmp/dsv4_ffn_dump.f3ZQ89`;
- exact checkpoint layer-20 TP4 attention and shared-expert block-FP8 slices,
  dequantized to the same cached BF16 representation used by production;
- current unified sparse paged-decode Triton kernel at TP4 H16/D512;
- production M32 `wo_a` einsum fallback and BF16 `wo_b` projection shape;
- current A4/R2/W8 grouped routed gate/up, group-32 INT8 quant, down and
  fixed-slot reduction kernels;
- a real diverse M32 route from pass 37/layer 20 of
  `/tmp/expert_distribution_recorder_1787803355.1855972.pt`, with 112 A4
  expert blocks;
- real TP4 shared-expert gate/up/down projection shapes and bounded SwiGLU.

The two arms are:

```text
serial:   attention subchain -> routed/shared MoE subchain
overlap:  attention subchain || routed/shared MoE subchain on distinct streams
```

This is deliberately an optimistic continuation gate.  It excludes both TP4
all-reduces, cross-layer graph state, dynamic page-table/indexer metadata and
outer scheduler/graph events.  The sparse core uses realistic contiguous
paged BF16 KV at fixed context lengths rather than mutating the service KV
pool.  It also excludes the MHC boundaries common to the full stages.  Adding
the omitted communication and state management cannot repair a failure caused
by GPU compute/cache contention here.

GPU availability was checked with `amd-smi process` before every valid run.
Physical GPU 0 had only the two BIO processes' sub-MiB, zero-GFX contexts; BIO
work on GPUs 4/5 was not touched.  Every timing below is seven-round ABBA with
30 timed repetitions per sample and a one-sample trim at each tail.

## Results

| KV rows/request | attention | MoE | serial | overlap | saving | overlap efficiency | continue |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 457.983 us | 549.269 us | 862.765 us | 801.841 us | 7.061% | 19.434% | no |
| 512 | 466.806 us | 548.454 us | 906.164 us | 793.631 us | 12.419% | 31.459% | no |
| 1024 | 495.552 us | 547.826 us | 1006.824 us | 839.083 us | 16.660% | 36.545% | no |

`overlap efficiency` is the realized saving divided by the theoretical saving
if the shorter subchain were hidden completely.  The attention output, routed
output and shared-expert output were bitwise exact between serial and overlap
for every accepted run.  No workspace race was observed.

Raw stdout:

```text
/tmp/dsv4_tp4_m32_attn_moe_overlap_oracle.log
/tmp/dsv4_tp4_m32_attn_moe_overlap_oracle_ctx512.log
/tmp/dsv4_tp4_m32_attn_moe_overlap_oracle_ctx1024.log
```

## Decision

Reject the TP4 M32 attention--MoE cross-layer wavefront and do not implement
the four-rank/two-layer decode graph.  Even the collective-free optimistic
oracle misses the predeclared 20% continuation gate at all three context
lengths.  Its best result is 16.66% at context 1024; adding two TP collectives,
fixed four-rank collective ordering, separate graph metadata and MHC state
would raise rather than lower the pair latency.  At the short/mid contexts
used by the current diverse decode benchmark, only 7.1--12.4% of serial time
is removed.

The result agrees with the older real TP8/M32 stage probe: attention projection
and paged-KV traffic compete with routed/shared expert weights for HBM, L2 and
CU scheduling.  Splitting M64 into two M32 graphs also gives up the accepted
whole-M64 routed work decomposition.  Continue optimizing the single M64
graph rather than adding decode-TBO state, communicators, or scheduler glue.
