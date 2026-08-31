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

Artifacts: `/tmp/dsv4_m128_prerouter_b.json`,
`/tmp/dsv4_m128_dpp_a.json`, `/tmp/dsv4_m128_prerouter_b2.json`.
