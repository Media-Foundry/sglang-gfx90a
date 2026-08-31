# Strict DSpark M64/M96 routed geometry sweep (rejected)

Date: 2026-09-01

## Scope

- Original DeepSeek-V4-Flash weights, TP4 local expert width 512.
- Standalone routed-stage oracle only; no production selector changed.
- Physical GPU 4 only, after `amd-smi process` reported no active process.
- Routes came from the strict gamma-three, anchor-only-disabled recorder made
  from 32 heterogeneous code requests.
- Both tiers retained A4/R2, the packed-FP4 LDS decoder, group-32 INT8
  intermediate quantization, FP32 partials, and fixed-order BF16 reduction.
- The accepted `G2080/D832/W8` geometry was compared with independent gate
  grids 1248/1664/2496/3120, down grids 624/1040/1248, and a four-wave arm.

The generalized oracle is
`scripts/rocm/bench_dsv4_dspark_m128_geometry.py --tokens {64,96}`.

## Correctness

Every candidate in both tiers passed:

- 100 activation/router-weight mutations with every visible intermediate and
  final output bitwise equal;
- 1000 HIP Graph replays with bitwise-stable output;
- seven-round symmetric timing.

## M96

The selected layer-20 route had 162 active experts, 220 A4 scans and maximum
occupancy 19. The accepted decode geometry measured 1043.218 us.

| candidate | complete stage (us) | versus current |
|---|---:|---:|
| G2496 / D832 | 1036.307 | +0.667% |
| G3120 / D832 | 1040.077 | +0.302% |
| G1664 / D832 | 1050.093 | -0.655% |
| G1248 / D832 | 1059.493 | -1.536% |
| G2080 / D624 | 1100.583 | -5.212% |
| G2080 / D1040 | 1087.238 | -4.049% |
| G2080 / D1248 | 1066.447 | -2.178% |
| four waves | 1041.203 | +0.194% |

The best saving was only 6.912 us for one routed layer, far below the
standalone continuation threshold and too small to survive service overlap.

## M64

The selected layer-20 route was substantially more concentrated than the old
native-M64 tuning route: 36 active experts, 114 A4 scans and maximum occupancy
50. The accepted decode geometry measured 618.918 us.

| candidate | complete stage (us) | versus current |
|---|---:|---:|
| G3120 / D832 | 617.943 | +0.158% |
| G1248 / D832 | 619.237 | -0.051% |
| G2496 / D832 | 620.027 | -0.179% |
| G1664 / D832 | 624.803 | -0.942% |
| G2080 / D624 | 647.411 | -4.401% |
| G2080 / D1040 | 643.735 | -3.855% |
| G2080 / D1248 | 619.236 | -0.051% |
| four waves | 621.986 | -0.493% |

The old M64 `G2080/D832/W8` geometry remains locally optimal even under the
strict DSpark concentrated route. Do not add a strict-tier geometry selector.

## Related closure

The existing signed-INT8 W2 hot-cache path was also re-audited and remains
closed. Prior mixed-cache latency regressed monotonically as hot-block coverage
rose from 47.4% to 57.9%; full TP4 INT8 and exact int5 repacks regressed 43.8%
and 21.3%, respectively. The strict-M64 held-out scan hit of roughly 76% would
send more blocks down the already slower double-byte path and does not justify
another GPU run.

Raw logs:

```text
/tmp/dsv4_strict_m64_geometry.log
/tmp/dsv4_strict_m96_geometry.log
```

