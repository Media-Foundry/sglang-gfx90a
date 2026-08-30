# DSV4 TP4 M64 grouped-FP4 geometry oracle

Date: 2026-08-30

## Real route and correctness corpus

A TP4/EP1 native-AR service captured tiers 1/32/64 and recorded 64 distinct
token-ID requests from `dsv4_tp8_diverse_64_input_ids.json`. The collection
used 16 warm decode passes, 32 recorded passes, and 8 tail passes. All 64
requests produced exactly 56 tokens and the France first-nine-token oracle was
exact. The accepted recorder is:

```text
/tmp/expert_distribution_recorder_1788072257.651073.pt
```

Every warm decode layer records 1536 global assignments. Dividing by the TP4
recorder replication gives exactly `64 * topk6 = 384` assignments.

The occupancy collector now accepts an explicit request count, and the grouped
oracle accepts an explicit batch size and recorded world size. The shared
deterministic count reconstruction remains duplicate-free and defaults to the
old M32 behavior.

## Gate grid

Fixed A4/R2/W8/LDS and down D832 on pass 20/layer 34:

| gate grid | full stage median | exact vs G2080 |
|---:|---:|---:|
| 832 | 1076.740 us | no |
| 1248 | 1065.790 us | no |
| 1664 | 1061.646 us | no |
| 2080 | 1060.810--1061.518 us | yes |
| 2496 | 1084.506 us | no |
| 3120 | 1082.266 us | no |
| 4160 | 1071.604 us | no |

The non-exact candidates have very large final-output errors (roughly
15k--22k max absolute), not harmless rounding differences. The current grid is
part of the fixed task-coverage mapping rather than a freely tunable occupancy
knob. Apparent timing wins from changing it are invalid.

## Down grid

Fixed gate G2080 and A4/R2/W8/LDS:

| down grid | full stage median | exact vs D832 |
|---:|---:|---:|
| 624 | 1075.539 us | no |
| 832 | 1062.381 us | yes |
| 1040 | 1057.100 us | no |
| 1248 | 1053.328 us | no |
| 1664 | 1055.728 us | no |

D832 is likewise structural. No geometry-only change is accepted.

## A8 closure

A8/R1/W4 reduced the real route from 174 A4 scans to 153 A8 scans, but the
complete stage regressed from 1046.245 us to 1726--1814 us for the tested
B832/B1248/B1664/B2080 variants. The candidates were also not exact against
production A4. A8 is rejected for TP4 M64.

## Production stage budget

Seven-round isolated medians for exact A4/R2/W8/G2080/D832 are:

| stage | median |
|---|---:|
| gate/up | 581.222 us |
| intermediate INT8 quantization | 38.498 us |
| down | 461.453 us |
| fixed reduction | 4.877 us |
| full routed stage | 1063.246 us |

Gate and down dominate. Quantization and final reduction cannot supply the
roughly 410 us/layer reduction implied by the 1300 tok/s target.

## Occupancy evidence

Across warm passes 16--47 and all 43 layers:

```text
active experts: median 146, p10/p90 128/165
A4 scans:       median 174, p10/p90 164/186
A4 padding:     median 44.83%, p10/p90 41.46%/48.39%
```

Pass 20/layer 34 has 61 experts with run length 1 and 36 with run length 2.
The first three hash-router layers exceed 51% A4 padding. This justifies a
single-launch occupancy-aware A1/A2/A4 decomposition, but not the previously
rejected multiple-bucket-launch design.

## Decision

- Keep production G2080/D832 and A4.
- Treat grid block counts as structural correctness parameters until the task
  mapping itself is redesigned.
- Next prototype: one static-buffer launch that selects A1/A2/A4 work per
  expert while preserving vectorized loads and exact fixed-slot reduction.
