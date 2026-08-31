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

## Corrected strict oracle result

The oracle used 100 mutated inputs/weights, 1000 HIP Graph replays, and ABBA
timing on one otherwise-idle gfx90a GCD.

| Shape | current | candidate | speedup | mutation max abs | max relative L2 | graph replay max abs |
|---|---:|---:|---:|---:|---:|---:|
| M128 N2048 K4096 | 58.572 us | 48.086 us | 1.2181x | 1 BF16 | 6.96e-5 | 0 |
| M128 N256 K4096 | 42.340 us | 28.435 us | 1.4890x | 1 BF16 | 1.06e-4 | 0 |

Neither candidate was bitwise equal to the current path on any of the 100
mutations, but both were stable over 1000 graph replays after correcting the
oracle to launch the graph once before cloning its reference output.  The
first run's large replay deltas were an oracle bug caused by cloning
uninitialized capture storage and are superseded by the table above.

## Service ABBA

The candidate was wired behind an independent `DSPARK` algorithm guard plus
the exact M128 and weight-shape guards. Native AR was unreachable. Both
services used physical GCDs 4-7, TP4/EP1/no-A2A, gamma 3, and 32 heterogeneous
code requests of 256 output tokens each.

- candidate: 929.905 / 911.391 / 909.720 tok/s; median 911.391
- control confirmation: 1045.539 / 974.480 / 1116.262 tok/s; median 1045.539
- candidate delta versus control median: -12.83%

All requests completed exactly 256 tokens with `finish=length`. The France
probe retained completion hash `73fbc3570829b132` and semantic Paris output.
The candidate therefore passed the service correctness screen but failed
performance decisively. The likely explanation is that a faster standalone
hipBLASLt GEMM consumes more CU or changes launch ordering enough to damage
the existing compressor/indexer/attention overlap.

## Decision

Reject and remove the production selector. Keep only the standalone oracle
and this record. Never add these entries to a global tuned CSV because that
could affect an AR M128 workload. This is another concrete example of a
microkernel win losing E2E because DSpark M128 relies on cross-branch overlap.
