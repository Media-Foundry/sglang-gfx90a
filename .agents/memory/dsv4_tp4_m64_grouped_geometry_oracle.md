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

## M64 logical W2-scale and row-prefetch oracle

The existing M32 logical-scale oracle was generalized to accept runtime batch
size and recorder world size, then run on the same real M64 pass20/layer34
route. It compares the CK-shuffled scale, logical down-only scale, and R2-packed
logical down scale using the exact grouped row-prefetch kernel.

All four layouts passed 100 activation/router-weight mutations bitwise at gate
BF16, quantized INT8 value/scale, down FP32 partial and final BF16 output.
Seven-round trimmed results were:

| stage | shuffled | logical down | R2-packed logical down |
|---|---:|---:|---:|
| gate | 406.005 us | 405.816 us | 405.997 us |
| quant | 46.456 us | 46.835 us | 46.186 us |
| down | 299.809 us | 279.496 us | 279.634 us |
| reduce | 5.349 us | 5.399 us | 5.257 us |
| full | 724.681 us | 703.627 us | 703.213 us |

Ordinary logical-down improves down by 7.27% and full routed by 2.99%.
R2-packing changes full time by only another 0.414 us, so it is not worth a
second layout/protocol. A down-only logical cache costs 16 MiB/layer or about
688 MiB/GCD across 43 layers. This passes the 3% routed continuation threshold
within rounding and proceeds to a strict M64 production/service experiment;
it is not yet a delivered default.

### Logical-down production checkpoint

Production wiring clones the checkpoint-order E8M0 W2 scale before AIter's
shuffle when the M64 selector is enabled. The runner requires the already
accepted strict M64 gate-DPP shape, W2 `[256,4096,256]`, A4/R2/W8/D832/LDS,
and then selects the exact row-prefetch logical-scale down specialization.
The wrapper permits M32 or M64 only; every other tier remains on shuffled
scales and the generic down kernel.

Graph tiers 1/32/64 captured successfully. The fixed 64-row teacher-forced
oracle matched the accepted baseline JSON-exactly for all output IDs, output
token-logprob rows and complete top-5 rows. Both real-request runs completed
64/64 requests at 256 tokens with `finish=length` and the France oracle exact.

Two independent services measured:

| service | HTTP resident | scheduler/model | mean step |
|---:|---:|---:|---:|
| 1 | 985.429 tok/s | 993.524 tok/s | 64.417 ms |
| 2 | 981.386 tok/s | 989.317 tok/s | 64.691 ms |
| center | **983.408 tok/s** | **991.421 tok/s** | about 64.55 ms |

Relative to the accepted M64 gate-DPP center `966.622/974.919 tok/s`, the
gain is approximately +1.74% HTTP and +1.69% scheduler.

The down-only cache costs about 688 MiB/GCD. With the same requested
`MAX_TOTAL_TOKENS=65536`, the server's admitted pool falls from 65536 to 61696,
a loss of 3840 tokens or 5.86%. Enable the selector by default in the explicit
TP4 speed profile to advance the throughput target; set
`SGLANG_DSV4_GFX90A_M64_LOGICAL_DOWN_SCALE=0` for the maximum-context profile.

## M64 occupancy-bucket closure

The existing exact A1/A2/A4 oracle was generalized to M64 and run on the real
pass20/layer34 route. The 384 assignments split into 61 A1 expert blocks, 36
A2 blocks and 77 A4-rest blocks. All five geometry profiles were bitwise exact
at gate intermediate, FP32 partial and final output; the best profile also
passed 100 mutated-input exactness checks.

Nevertheless every multi-launch bucket profile regressed the complete routed
stage:

| profile | baseline | bucket | regression |
|---|---:|---:|---:|
| gate/down 832/416/416 | 426.008 us | 459.061 us | +7.76% |
| smaller A2/A4 gate grids | 426.392 us | 465.008 us | +9.06% |
| smallest tested grids | 425.933 us | 541.549 us | +27.14% |

For the best profile, gate changed `224.483→249.851 us` and down
`186.605→193.211 us`. Separate bucket launches and reduced latency hiding cost
more than the smaller A1/A2 accumulator templates save. Do not connect the
multi-launch occupancy sorter to production at M64. Any follow-up must remain
one GPU launch or materially change weight reuse rather than repeating bucket
launches.

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

## Native HIP MHC pre-mix M64 geometry (2026-08-30)

The TP4/EP1 path was audited before changing MHC. Contrary to the earlier
working assumption, M64 does not fall back to AIter/Triton: the explicit TP4
profile already reaches the native gfx90a wave64 pre-mix. Production used the
historical three output rows per CTA geometry, originally selected under the
lower-batch/Mori workload.

An isolated native-HIP oracle swept `1/2/3/4/6/8` output rows per CTA at
`T=64`, retaining the exact wave64 reduction order within every row. All six
variants matched the rows=3 reference bitwise, and the complete sweep passed
100 random activation/weight mutations bitwise. Seven symmetric timing rounds
gave:

| rows/CTA | median | trimmed |
|---:|---:|---:|
| 1 | 45.970 us | 45.967 us |
| 2 | **37.686 us** | **37.685 us** |
| 3 | 39.450 us | 39.450 us |
| 4 | 48.907 us | 48.911 us |
| 6 | 45.340 us | 45.339 us |
| 8 | 50.684 us | 50.687 us |

Rows=2 is 4.47% faster than rows=3 in isolation. Production therefore selects
rows=2 only when the runtime token count is exactly 64; every other graph tier
retains rows=3. Graph tiers 1/32/64 captured successfully. A fixed 64-row
teacher-forced request matched the preceding logical-W2 checkpoint JSON-exact
for all output IDs, token logprobs, and complete top-5 rows.

Two same-service real-diverse rounds measured resident throughput of
`984.257/983.556 tok/s`, center `983.907 tok/s`. The second round's scheduler
rate was `992.019 tok/s` with mean step `64.515 ms`. Against the preceding
independent-service centers `983.408/991.421 tok/s`, this is only about
`+0.05/+0.06%`, below a meaningful service checkpoint. Both rounds completed
64/64 requests at 256 tokens with `finish=length` and the France oracle exact.
Keep the exact M64 selector as a non-regressing small specialization, but do
not attribute a material end-to-end gain to it or spend another service cycle
on nearby MHC geometry.

## M64 AIter BF16 tuned-GEMM closure (2026-08-30)

The service startup log reported untuned BF16 shapes `(M,N,K) =
(64,{512,1024,2048},4096)` and selected AIter's torch fallback. An exhaustive
gfx90a tuner pass enumerated about 2200 hipBLASLt solutions per shape. An
independent seven-round CUDA-Graph ABBA check, rather than the tuner's profiler
ranking, measured:

| N | torch graph | hipBLASLt graph | micro speedup | solution |
|---:|---:|---:|---:|---:|
| 512 | 28.960 us | 21.713 us | 33.37% | 3963 |
| 1024 | 33.670 us | 24.600 us | 36.87% | 3990 |
| 2048 | 44.022 us | 37.578 us | 17.15% | 5087 |

All candidates were finite and bitwise stable across repeated graph replays.
Their relative L2 difference from torch was `3.49e-5--8.28e-5`, reflecting a
different valid BF16 accumulation order. The fixed 64-row full-model
teacher-forced oracle nevertheless matched the accepted checkpoint JSON-exact
for output IDs, token logprobs, and complete top-5 rows.

The real-diverse service result did not inherit the micro win. Two resident
rounds were `984.076/983.614 tok/s` (center `983.845`), while the adjacent
rows=2 baseline center was `983.907 tok/s`. Scheduler/model timing was
`992.156 tok/s` versus `992.019 tok/s`, also noise. Every request completed at
256 tokens with `finish=length` and the France oracle exact. These GEMMs are
hidden by the current side-stream schedule or are not rank-max critical.
Do not add the tuned rows to the production AIter config merely because their
isolated kernels are faster; prioritize routed FP4 and attention consumer
boundaries instead.
