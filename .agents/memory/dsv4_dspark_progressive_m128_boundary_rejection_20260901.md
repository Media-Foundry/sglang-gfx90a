# DSpark progressive M128 boundary rejection (2026-09-01)

## Scope

This experiment tested a DSpark-only TP4 optimization for the gamma-three
M128 boundary.  It did not change native AR, model weights, or attention
semantics.  All GPU measurements used physical GCDs 4--7 after an `amd-smi`
process check.

The production boundary performs one exact BF16 M128 AIter two-stage
all-reduce followed by the M128 fused MHC post/pre+RMS path.  The candidate
split the request-major candidate rows into:

- draft rows: M96 (candidate lanes 1--3), reduced and processed on a side stream;
- anchor rows: M32 (candidate lane 0), reduced and processed after routed MoE.

## Exact semantic reduction

Two ordinary compact M96/M32 two-stage collectives are not equivalent to the
production M128 collective.  M128 partitions the flattened tensor into four
32-row owner quarters and accumulates ranks in owner-rotated order.  Compact
collectives change that owner mapping and produced differences up to 11.0.

A temporary AIter diagnostic kernel therefore mapped every compact row back to
its original M128 row, derived the owner from `original_row // 32`, accumulated
the four peers in the original owner-rotated order, and downcast once to BF16.
It passed:

- 100 rank-distinct input mutations, bitwise exact;
- 1000 HIP graph replays, bitwise exact;
- maximum absolute error 0.

With production `AITER_GFX90A_AR_1M_BLOCKS=12`, rank-max component medians were:

| Component | Time (us) |
|---|---:|
| production full M128 all-reduce | 48.026 |
| direct draft M96 reduction | 67.530 |
| direct anchor M32 reduction | 31.570 |
| sequential direct M96+M32 | 94.662 |

The sequential form is therefore rejected.

## MHC shape measurements

The same fused MHC path on one gfx90a GCD measured:

| Rows | Median (us) |
|---:|---:|
| 32 | 38.992 |
| 96 | 85.147 |
| 128 | 108.890 |

Ignoring resource contention, fully hiding the M96 branch would reduce the
exposed boundary from roughly `48.0 + 108.9 = 156.9 us` to
`31.6 + 39.0 = 70.6 us`, an apparent upper bound of about 86 us/layer.

## Multi-stream graph result

The decisive oracle used three independent AIter communicators so draft and
anchor reductions could execute concurrently without sharing signal state.  A
main-stream chain of repeated real M32 MHC kernels represented the routed
critical section.  The candidate started draft M96 reduction+MHC on a side
stream, ran the main chain, then performed anchor M32 reduction+MHC and joined.

It again passed 100 mutations and 1000 graph replays bitwise exactly, but ABBA
rank-max timing rejected the optimization:

| Main-chain proxy | Baseline (us) | Progressive (us) | Delta |
|---|---:|---:|---:|
| 9 x M32 MHC | 492.388 | 514.514 | -22.126 us (-4.30%) |
| 16 x M32 MHC | 738.902 | 780.751 | -41.849 us (-5.36%) |

The regression grows with the overlap window.  The draft branch is not free:
its custom all-reduce and M96 MHC contend with the routed branch for CU,
cache/HBM, and peer progress.  The arithmetic critical-path upper bound is
therefore not realizable with ordinary concurrent kernels on gfx90a.

## Decision

- Do not connect this path to the service.
- Do not reuse the temporary semantic-row AIter primitive in production.
- Restore the previous production AIter module and remove the diagnostic source.
- Any future revisit requires a single cooperative boundary kernel/protocol
  that reserves resources or progresses rows without launching a competing
  M96 compute grid.  Merely moving draft AR+MHC to another stream is disproved.

The accepted E2E checkpoint remains approximately 1.56--1.57k tok/s at BS32
on the fixed real-varied workload (best observed round approximately 1.60k).
