# DSpark M128 compressor hipBLASLt rejection (2026-09-01)

## Scope

This experiment targeted only the DeepSeek-V4 DSpark TARGET_VERIFY M128
compressor projections.  It did not change production dispatch, and no AR
path was modified.

The two exact BF16 shapes were:

- core C4 compressor: `M=128, N=2048, K=4096`
- indexer C4 compressor: `M=128, N=256, K=4096`

The production AIter `tgemm.mm` had no tuned entry for either shape and fell
back to its default torch solution.

## Tuning

The AIter hipBLASLt tuner on physical GPU 4 selected:

| Shape | Solution | tuner time | tuner err_ratio |
|---|---:|---:|---:|
| M128 N2048 K4096 | 4129 | 43.3418 us | 0.0 |
| M128 N256 K4096 | 5097 | 18.7049 us | 0.0 |

The standalone oracle is
`scripts/rocm/bench_dsv4_dspark_m128_compressor_hipblaslt.py`.

## Strict oracle result

The oracle used 100 mutated inputs/weights, 1000 HIP Graph replays, and ABBA
timing on one otherwise-idle gfx90a GCD.

| Shape | current | candidate | speedup | mutation max abs | max relative L2 | graph replay max abs |
|---|---:|---:|---:|---:|---:|---:|
| M128 N2048 K4096 | 59.381 us | 49.358 us | 1.2031x | 1 BF16 | 6.96e-5 | 306 |
| M128 N256 K4096 | 42.083 us | 28.014 us | 1.5022x | 1 BF16 | 1.06e-4 | 272 |

Neither candidate was bitwise equal to the current path on any of the 100
mutations.  More importantly, both explicit hipBLASLt solutions were grossly
unstable under graph replay: the same captured graph and unchanged tensors
produced maximum absolute output changes of 306 and 272.  This is not a
minor reduction-order difference and blocks service integration.

## Decision

Reject both explicit hipBLASLt solutions for the captured DSpark service.
Do not add them to the global AIter tuned CSV and do not dispatch them from
`linear_bf16_fp32`; doing so could also affect an AR M128 workload.  The
microbenchmark speedup is real in eager execution, but the graph correctness
failure makes an E2E France-only check insufficient.

If revisited, first investigate why `hipb_mm` solution replay is graph-unsafe
on gfx90a (workspace lifetime, capture support, or solution-specific state).
No service benchmark should be run until a 1000-replay fixed-input oracle is
stable.
