# DSV4 TP4 DSpark M51 routed specialization service rejection

Date: 2026-08-31

## Scope

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- Original DeepSeek-V4-Flash weights, TP4 / EP1 / no A2A
- DSpark block size 2, forced verify-budget fraction 0.30
- Static ragged verify, CUDA graph tiers 1 through 32
- 32 distinct concrete coding prompts, 256 generated tokens each
- Independent-service A/B/B/A with the official France first-nine-token oracle
- Static memory fraction 0.90; the original 0.80 profile was insufficient for
  the 32,768-token pool plus all DSpark graph tiers and logical W2-scale caches

## Real M51 route and isolated oracle

The target-only recorder found a representative layer-20 M51 route with 306
assignments, 129 active experts, 146 A4 scans and maximum occupancy 10.  The
candidate retained original packed FP4 weights and the fixed-order FP32 output
reduction, while selecting:

```text
gate: A4 / R2 / G1664 / LDS decode / DPP / row prefetch
down: A4 / R2 / D832 / LDS decode / logical W2 scale / row prefetch / W4
```

Seven-round timing on physical GCD 4:

| arm | complete routed stage | gate | quant | down | reduce |
|---|---:|---:|---:|---:|---:|
| current LDS | 636.617 us | 370.963 us | 19.131 us | 251.113 us | 5.451 us |
| M51 candidate | 569.801 us | 336.819 us | 19.002 us | 218.078 us | 4.859 us |

The complete routed stage improved by 10.5%.  One hundred route/input
mutations were elementwise exact and 1,000 HIP Graph replays were bitwise
stable.

## Production integration

`SGLANG_DSV4_GFX90A_DSPARK_M51_ROUTED_SPECIALIZATION` is default-off and has
strict model/shape/geometry guards.  It is accepted only for M51, top-6,
E256, TP4 K4096/I512, A4/R2, LDS decode, G1664/D832 and W4 down.  Other tiers
retain their existing kernels.  Logical W2 scales are cached at load time only
when the switch is explicitly enabled.

The first service capture exposed a fail-fast wrapper assertion that still
allowed only M32/M64.  The wrapper was narrowed to accept M51 only with G1664
and W4; it does not generally admit arbitrary M or geometry.

## Real-code service ABBA

Each arm used a fresh service.  Every service passed the exact France prefix
and semantic Paris check.  Every coding request generated exactly 256 tokens
and finished with `length`.

| arm | resident samples (tok/s) | aggregate samples (tok/s) | scheduler samples (tok/s) | host step (ms) |
|---|---|---|---|---|
| A1 current | 719.264, 722.171 | 630.141, 630.339 | 684.021 | 88.182 |
| B1 M51 | 729.171, 723.610 | 629.538, 339.577 | 687.836, 721.799 | 89.132, 93.199 |
| B2 M51 | 708.730, 708.031 | 615.617, 617.891 | 675.938, 665.310 | 90.679, 89.690 |
| A2 current | 685.749, 698.860 | 583.200, 619.601 | 657.981, 681.188 | 93.108, 93.355 |

Median centers over the available per-round samples:

```text
resident:  709.062 -> 716.170 tok/s  (+1.00%)
scheduler: 681.188 -> 681.887 tok/s  (+0.10%)
host step:  93.108 ->  90.184 ms     (-3.14%)
aggregate: 624.871 -> 616.754 tok/s  (-1.30%)
accept:      2.128 ->   2.137        (+0.40%)
```

The isolated win does not survive as a material service-level improvement.
Keep the exact specialization and benchmark as reusable research machinery,
but do not enable it in the shipped TP4 BS32 profile.

## Artifacts

```text
/tmp/dsv4_dspark_m51_route_legacy.pt
/tmp/dsv4_dspark_m51_specialization_production_recheck.json
/tmp/dsv4_dspark_m51_{A1,B1,B2,A2}.json
/tmp/dsv4_dspark_m51_{A1,B1,B2,A2}_france.json
/tmp/dsv4_m51_{A1,B1,B2,A2}.log
```
