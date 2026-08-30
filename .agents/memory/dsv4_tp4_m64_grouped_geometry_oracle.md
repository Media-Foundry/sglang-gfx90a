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

## Initial oracle defect and correction

The first geometry run exposed a test-only metadata bug: `make_metadata()`
used the historical module constant `M=32` as its padding sentinel. At M64,
token 32 is valid, so padded assignments raced with real writes and made every
profile non-self-exact. Production AIter sorting was not affected. The helper
now derives the invalid sentinel from `topk_ids.shape[0]`; all geometry results
below must be regenerated after this correction. The pre-correction timing and
cross-profile exactness tables are retained only as provenance and must not be
used to select a grid.

## Pre-correction gate grid (invalid for selection)

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

These non-exact results came from the M32 padding-sentinel alias, not from a
proven structural grid constraint.

## Pre-correction down grid (invalid for selection)

Fixed gate G2080 and A4/R2/W8/LDS:

| down grid | full stage median | exact vs D832 |
|---:|---:|---:|
| 624 | 1075.539 us | no |
| 832 | 1062.381 us | yes |
| 1040 | 1057.100 us | no |
| 1248 | 1053.328 us | no |
| 1664 | 1055.728 us | no |

These exactness results likewise require regeneration.

## Pre-correction A8 screen (timing indicative, correctness invalid)

A8/R1/W4 reduced the real route from 174 A4 scans to 153 A8 scans, but the
complete stage regressed from 1046.245 us to 1726--1814 us for the tested
B832/B1248/B1664/B2080 variants. The candidates were also not exact against
production A4. A8 is rejected for TP4 M64.

## Corrected geometry results

After deriving the padding sentinel from runtime M, every tested A4 grid was
repeat-self-exact and cross-profile bitwise exact. The grid-stride kernels
therefore behave as their source indicates; block count is not a mathematical
coverage constraint.

Corrected gate sweep with down D832:

| gate grid | full stage median |
|---:|---:|
| 832 | 770.292 us |
| 1248 | 769.793 us |
| 1664 | 760.782 us |
| 2080 | **752.980 us** |
| 2496 | 758.449 us |
| 3120 | 759.083 us |
| 4160 | 760.542 us |

Corrected down sweep with gate G2080:

| down grid | full stage median |
|---:|---:|
| 624 | 801.598 us |
| 832 | **753.768 us** |
| 1040 | 770.225 us |
| 1248 | 770.171 us |
| 1664 | 749.575 us |

The isolated D1664 result appeared 0.56% faster, but an adjacent two-profile
repeat measured D832/D1664 at `752.052/752.572 us`; it did not reproduce.
Production remains G2080/D832.

## Production stage budget

The pre-correction table below is invalid because padded assignments aliased a
real M64 token. It is retained only to explain the original diagnosis:

| stage | median |
|---|---:|
| gate/up | 581.222 us |
| intermediate INT8 quantization | 38.498 us |
| down | 461.453 us |
| fixed reduction | 4.877 us |
| full routed stage | 1063.246 us |

Corrected adjacent medians for exact A4/R2/W8/G2080/D832 are:

```text
gate/up                 425.225 us
intermediate INT8 quant  42.091 us
down                    304.117 us
fixed reduction           4.939 us
full routed stage       752.052 us
```

Production AIter sorting already used a correct invalid sentinel and never had
the test-only race.

## BS64 service marker budget

A marker-only service generated 128 tokens for all 64 distinct requests, with
every finish reason `length` and the France oracle exact. Across 16 four-rank
groups, layer-20 rank-max medians were:

```text
attention-entry MHC        91.76 us
attention prepare         245.92 us
attention core             74.40 us
attention output          163.68 us
FFN-entry MHC             100.32 us
MoE coarse                931.12 us
router                     46.56 us
top-k                      16.16 us
routed experts            805.92 us
TP4 all-reduce tail        64.80 us
```

Its roughly 946 tok/s resident value is not a formal performance result because
the markers add work; the trace is used only for localization.

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

- Keep production G2080/D832 and A4; corrected grid sweeps confirm both.
- Treat all pre-correction exactness and component timing as invalid provenance.
- Then prototype one static-buffer launch that selects A1/A2/A4 work per
  expert while preserving vectorized loads and exact fixed-slot reduction.
