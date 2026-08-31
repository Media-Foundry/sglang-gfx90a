# DSpark M128 pre-router anchor compaction candidate (2026-09-01)

The earlier post-router M32 compaction mixed a faster routed kernel with costly
service seams. This experiment moves anchor extraction before router/TopK:
only the 32 exact anchor rows execute router, TopK and routed MoE, while the
shared expert and all other target paths remain physical M128. The selector is
downstream of the existing gfx90a TARGET_VERIFY/BS32/width-4/M128 parent guard,
so native AR is unreachable.

TP4/EP1, original weights, GCDs 4--7, 32 distinct code requests, 256 tokens,
stream interval one:

```text
B1: 968.379 / 959.651 / 965.719, median 965.719 tok/s
A:  887.333 / 915.591 / 946.398, median 915.591 tok/s
B2: 1005.363 / 904.674 / 848.586, median 904.674 tok/s
center(B1,B2): 935.197 tok/s, +2.14% versus A
```

All nine rounds retained the exact France first nine tokens, semantic Paris,
and 32x256 `finish=length`. B2's throughput fall tracked accepted length
(`2.605 -> 2.331 -> 2.264`), so the first 1005 result is not a stable kernel
checkpoint. Host-step samples indicate only roughly 2--3% structural benefit.

Decision: retain the strict experimental implementation but keep the profile
default off because the ABBA center does not clear the 5% checkpoint gate.
Future work should fuse anchor gather with quant/sort and avoid the full-output
zero/scatter seam before another service A/B.

Follow-up graph replay timing on physical GCD4 bounded those tensor seams:

```text
M128 -> M32 gather:       6.735 us
M32 -> M128 zero/scatter: 8.080 us
combined captured chain: 10.936 us/layer
```

Even deleting the combined chain entirely saves only about 0.47 ms per
43-layer target pass, well below the 40-us/layer continuation threshold.
Therefore do not build a complex strided carrier solely for these copies; the
remaining opportunity is router/TopK/attention/aux-logits work, not memcpy.

Artifacts: `/tmp/dsv4_m128_prerouter_b.json`,
`/tmp/dsv4_m128_dpp_a.json`, `/tmp/dsv4_m128_prerouter_b2.json`.

## Current-stack revalidation and promotion

The candidate was re-run after the strict target occupancy/graph fixes on the
same physical GCDs 4--7, with five rounds of 32 heterogeneous requests, 256
tokens each and stream interval one. The adjacent control was a separately
loaded service with the compact flag explicitly disabled.

| arm | resident rounds (tok/s) | median | trimmed mean | France semantic |
|---|---|---:|---:|---:|
| compact=1 | 1127.65 / 1119.62 / 1126.92 / 1139.34 / 1074.32 | **1126.92** | **1124.73** | 5/5 |
| compact=0 | 1095.87 / 1042.95 / 1168.34 / 1096.17 / 1074.50 | 1095.87 | 1088.85 | 2/5 |

The current-stack gain is +2.83% by median and +3.30% by trimmed mean. Every
request in both arms generated 256 tokens with `finish=length`; the control's
France failures make it unsuitable as a quality-approved checkpoint despite
being useful as a timing arm. The compact profile kept semantic Paris in all
five rounds. As expected for the already-declared anchor-only approximation,
cross-round completion hashes were not exact and are not represented as
strict target correctness.

Because this direction now improves both the resident critical path and the
observed semantic stability, the TP4 BS32 DSpark profile defaults
`SGLANG_DSV4_GFX90A_DSPARK_M128_PRE_ROUTER_COMPACT=1`. The model-side
TARGET_VERIFY/BS32/width-4/M128 parent guard remains the authoritative AR
negative guard.

Current artifacts:

```text
/tmp/dsv4_gamma3_prerouter_5r.json
/tmp/dsv4_gamma3_prerouter_control_allow_5r.json
```
