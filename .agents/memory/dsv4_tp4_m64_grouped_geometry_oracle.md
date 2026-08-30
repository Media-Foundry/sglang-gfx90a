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

## TP4 M64 C128 attention multistream checkpoint

Source audit showed that C4 M64 already uses the profile's multistream path,
while C128 layers never allocate HIP alternate streams. A strict default-off
selector was added for gfx90a, TP4, C128, decode M64 and graph capture only.
It reuses the established producer schedule: the core compressor runs on its
side stream and is joined before its consumer; Q/K math is unchanged.

The layer-21 baseline/candidate C128 prepare rank-max medians were
`168.32/156.24 us`, saving about `12.08 us` per C128 layer. Other coarse
intervals were unchanged within marker noise. The no-indexer C128 path also
now writes marker slot15 after the compressor join so the trace validator does
not read a stale C4 value.

Two no-marker candidate rounds with 64 distinct requests and 256 generated
tokens measured:

```text
HTTP common-resident 953.322 / 953.351 tok/s
center               953.337 tok/s
scheduler/model      961.454 tok/s
mean decode step      66.566 ms
```

Against the adjacent accepted BS64 baseline center `949.923 tok/s` and model
rate `956.989 tok/s`, this is approximately +0.36% HTTP and +0.47% scheduler.
Both rounds completed 64/64 requests at length 256 with `finish=length` and the
France oracle exact.

Because long autoregressive hashes drift even between repetitions of one
service, correctness used a fixed 64-row teacher-forced one-token oracle.
Baseline and candidate matched 64/64 output IDs, 64/64 output token-logprob
rows and 64/64 complete top-5 logprob rows JSON-exactly.

Enable `SGLANG_DSV4_GFX90A_TP4_M64_C128_ATTN_MULTISTREAM=1` by default inside
the explicit TP4 profile. The environment variable remains an exact rollback.

## M64 DPP reduction closure

The existing shuffle-versus-DPP A/G/D/B oracle was rerun at M64 on the real
pass20/layer34 route. One hundred mutated activation/router-weight cases were
bitwise exact at intermediate BF16, FP32 partial and final BF16 boundaries.
Gate-only and down-only candidates each passed 1000 captured replays exactly.

Seven-round ABBA trimmed full-stage centers were:

```text
A / shuffle reference  748.79--749.25 us
G / DPP gate only       734.82 us   (about -1.87%)
D / DPP down only       745.17 us   (about -0.55%)
B / DPP gate+down       731.40 us   (about -2.36%)
```

The combined candidate saves about 17.7 us/layer in isolation but fails the
predeclared 5% service-continuation gate. M32 history also showed that the DPP
micro win could worsen graph/service CU scheduling. Do not wire M64 DPP into
production from this result alone; the narrower gate-only follow-up below adds
the required service evidence.

### Gate-only production follow-up

Because gate-only DPP had previously shown a weak positive M32 service trend,
it was connected behind a separate strict gfx90a/TP4/M64/A4/R2/W8/G2080/LDS
selector. Down remains the production shuffle D832 kernel; the service-negative
combined DPP path is not enabled. The production wrapper's DPP contract was
expanded from M32 to M32-or-M64 while retaining all other exact shape guards.

The fixed 64-row teacher-forced comparison against the accepted baseline was
JSON-exact for every output ID, output token-logprob row and full top-5 row.
The standalone mutation and graph oracle had already passed 100 mutations and
1000 replays at intermediate BF16, FP32 partial and final BF16 boundaries.

Two independent candidate services, each running 64 distinct requests for 256
tokens, measured:

| service | HTTP resident | scheduler/model | mean step |
|---:|---:|---:|---:|
| 1 | 967.572 tok/s | 975.903 tok/s | 65.580 ms |
| 2 | 965.672 tok/s | 973.934 tok/s | 65.713 ms |
| center | **966.622 tok/s** | **974.919 tok/s** | about 65.65 ms |

Relative to the C128-overlap baseline `953.337/961.454 tok/s`, the independent
center improves by approximately 1.39% HTTP and 1.40% scheduler. Every request
completed at length 256 with `finish=length` and the France oracle exact.

Enable `SGLANG_DSV4_GFX90A_M64_DPP_GATE=1` by default only inside the explicit
TP4 profile. It remains an environment rollback and is unreachable at all
other graph tiers.

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
