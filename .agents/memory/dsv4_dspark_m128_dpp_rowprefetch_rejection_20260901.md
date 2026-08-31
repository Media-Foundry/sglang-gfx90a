# DSpark M128 DPP/row-prefetch rejection (2026-09-01)

## Scope

TP4/EP1, original DeepSeek-V4-Flash weights, physical GCDs 4--7, static
gamma-three DSpark, 32 distinct code requests and 256 output tokens. Native AR
was never eligible for the temporary selector.

## Exact routed-stage oracle

`scripts/rocm/bench_dsv4_dspark_m128_anchor_geometry.py` compares the current
sentinel-padded M128 routed stage with DPP gate and logical-scale row-prefetch
down tactics on a real layer-20 route. All arms passed 100 input mutations and
1000 HIP graph replays bitwise exactly.

```text
current M128                         482.597 us
M128 DPP gate                       476.638 us  (+1.25%)
M128 row-prefetch down              470.546 us  (+2.56%)
M128 DPP + row-prefetch W8          462.813 us  (+4.27%)
M128 DPP + row-prefetch W4          455.193 us  (+6.02%)
compact M32 DPP                     429.081 us (+12.47%)
```

Raw oracle: `/tmp/dsv4_m128_anchor_geometry_abglb4.json`.

## Service ABBA

The W4 combination was temporarily wired behind a DSpark-algorithm check and
exact M128/H4096/top-6/TP4 weight-shape guards. Three real-diverse rounds per
arm produced:

```text
candidate: 868.478 / 945.709 / 862.773, median 868.478 tok/s
control:   887.333 / 915.591 / 946.398, median 915.591 tok/s
```

The candidate regressed the median by about 5.15%. France retained the exact
first nine tokens and semantic Paris in all six rounds; every one of the
32x256 requests finished with `finish=length`.

## Decision

Reject production M128 DPP/row-prefetch wiring. The isolated kernel gain is
reversed by whole-graph CU/cache scheduling. The production selector and its
wrapper whitelist changes were removed. Retain the oracle because it also
proves that the earlier physical M32 compaction loss came from its service
seams rather than the compact routed kernel itself.

Service reports: `/tmp/dsv4_m128_dpp_b1.json` and
`/tmp/dsv4_m128_dpp_a.json`.
