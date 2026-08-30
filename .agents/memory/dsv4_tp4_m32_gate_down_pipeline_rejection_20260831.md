# TP4/M32 gate/down pipeline rejection (2026-08-31)

Two progressively finer producer decompositions were tested on physical GCD 4
before implementing a persistent gate/down macro-kernel.

## I32 producer geometry

The existing expert-owned exact DPP gate oracle was extended from owner CTA
fan-ins 1/4/8 to 16.  Owner16 maps each CTA to two W8/R2 tile groups, i.e. one
complete I32 group for one padded A4 expert block.  A synthetic route preserves
the measured diverse-request envelope: 192 assignments, 106 active experts,
and 113 padded A4 blocks.

All four geometries were bitwise exact for 100 activation mutations and their
device-scope publication counters/epochs were exact.

| Geometry | Gate latency (us) | Change vs G2080 |
|---|---:|---:|
| current G2080 | 244.462 | baseline |
| owner1 | 562.210 | +129.98% |
| owner4 | 296.679 | +21.36% |
| owner8 | 277.242 | +13.41% |
| owner16 / I32 | 269.838 | +10.38% |

Owner16 narrows but does not eliminate the producer decomposition/publication
tax.

## Two-chunk real gate/down overlap

The exact existing gate/down pipeline oracle was generalized to M32.  The 113
expert blocks were split 56/57 on an expert boundary.  The main stream ran the
two gate chunks while a side stream consumed completed chunks with the exact
group32 quant+down kernel.  Both chunks write disjoint expert slots into the
same FP32 partial buffer; the unchanged fixed-order reduction runs after the
join.  One hundred mutations were bitwise exact.

Seven-round ABBA results:

| gate blocks | down CTAs/expert | baseline (us) | pipeline (us) | change |
|---:|---:|---:|---:|---:|
| 1040 | 8 | 438.457 | 475.557 | +8.46% |
| 1040 | 12 | 438.374 | 476.773 | +8.76% |
| 1040 | 16 | 438.140 | 474.312 | +8.26% |
| 2080 | 8 | 438.238 | 476.765 | +8.79% |
| 2080 | 12 | 438.041 | 480.246 | +9.63% |
| 2080 | 16 | 438.148 | 476.779 | +8.82% |

## Decision

Do not implement the proposed persistent producer/consumer macro-kernel.  Both
its required producer geometry and a safer two-stream overlap upper-bound are
already slower.  Fine-grained ready/acquire polling would add the publication
and CU-residency costs previously measured in the owner/readiness oracles.
Revisit only if the gate or down arithmetic itself changes materially.
