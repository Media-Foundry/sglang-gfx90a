# DSV4 TP4 DSpark M84 BF16 GEMM tuning rejection

Date: 2026-08-31

## Scope

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- TP4 / EP1 / no A2A, original DeepSeek-V4-Flash weights
- DSpark compact verification, causal draft attention
- Forced verify budget fraction: 0.30
- CUDA graph request tiers: 1 through 32
- Workload: 32 distinct concrete coding prompts, 256 generated tokens
- Correctness: official France first-nine-token oracle after each service start

The TP4 BS32 profile had still overridden the repository-wide second-pair
default with `DEFAULT_GPUS=0,1,2,3`.  The profile now defaults to physical
GCDs 4--7; an explicit `HIP_VISIBLE_DEVICES` remains authoritative.

## Standalone tuning

The four BF16 target-verify GEMMs initially attributed to the M84 tier were
tuned with AIter's hipBLASLt tuner on physical GCD 4.  All selected solutions
reported zero tuner error:

| M | N | K | solution | time (us) |
|---:|---:|---:|---:|---:|
| 84 | 256 | 4096 | 3931 | 14.7704 |
| 84 | 512 | 4096 | 3930 | 22.0329 |
| 84 | 1024 | 4096 | 5087 | 27.8447 |
| 84 | 2048 | 4096 | 4009 | 40.2921 |

The input manifest is
`.agents/experiments/dsv4_dspark_m84_bf16_untuned.csv`.  Generated tuner
artifacts were kept under `/tmp/dsv4_dspark_m84_bf16_{best,profile}.csv`.

## Service A/B

Both services used the same physical GCDs, graph tiers, memory budget and
32-prompt manifest.  Round 0 includes warm-state and request-arrival effects;
round 1 is the comparable resident result.

| configuration | resident tok/s | scheduler tok/s | host step (ms) | accept length | France exact |
|---|---:|---:|---:|---:|---:|
| control | 401.877 | 397.169 | 82.916 | 1.910 | yes |
| M84 tuned | 401.525 | 397.796 | 82.775 | 1.974 | yes |

The change is within run noise: resident throughput changed by -0.09% and
scheduler throughput by +0.16%.  Do not enable the M84 table as a performance
default.

## Root cause of the null result

Compact verification does not execute a single fixed M84 BF16 shape.  After
startup, target rows varied with live draft survival and request state.  The
logs observed many exact row counts, including M13, M19, M26, M46, M69,
M91, M104, M109--M113.  AIter caches the fallback per exact shape, so an M84
entry only affects steps that happen to produce exactly 84 rows.

Consequently, the earlier M84 graph-tier label was not a sufficient runtime
shape oracle.  Future work should either:

1. measure a real post-warm row-count histogram and tune a small set of
   high-frequency dynamic-M buckets; or
2. deliberately pad the BF16 subpath to stable captured buckets and verify
   that the saved GEMM time exceeds the padding and scheduling cost.

## Artifacts

```text
/tmp/dsv4_dspark_m84_control_A1.log
/tmp/dsv4_dspark_m84_control_A1.json
/tmp/dsv4_dspark_m84_tuned_B1.log
/tmp/dsv4_dspark_m84_tuned_B1.json
/tmp/dsv4_dspark_m84_bf16_best.csv
/tmp/dsv4_dspark_m84_bf16_profile.csv
```
