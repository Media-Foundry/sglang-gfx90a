# DSV4 gfx90a block-M64 follow-up rejections (2026-09-02)

## Scope

- Original DeepSeek-V4-Flash weights, four physical gfx90a GCDs.
- Accepted large-prefill BF16-CK block-M64 V1 profile at commit `fe114fdebd`.
- Every GPU run was preceded by `amd-smi process --general --sort-by-pid`.

## Existing CK instance sweeps

All twelve generated FP32 stage-2 instances were tested at M27648 with the
accepted stage-1 kernel.  Every output was exact, but complete routed latency
only ranged from 30.946 to 31.354 ms.  The best change was below 1.5%, so no
stage-2 kernel override was retained.

All eleven generated BF16 stage-1 instances were then screened with block-M64.
Only the two V1 N64 instances were numerically valid.  The accepted K128
instance remained fastest (about 31.01 ms complete); K64 took about 43.58 ms.
All V3/N128 and mismatched N32/N128/N256 candidates produced large errors or
unwritten output and are rejected regardless of their apparent timing.

## W2 dequant overlap rejection

An event-ordered prototype ran W2 FP4-to-BF16 expansion on a dedicated stream
while the main stream sorted routes and executed stage 1.  Stage 2 waited on a
GPU event, with no CPU synchronization and unchanged reduction order.

Ten-sample A/B/B/A results:

| routing | sequential | overlap | result |
|---|---:|---:|---|
| balanced | 31.005 ms | 31.312 ms | 1.0% slower |
| skewed | 28.732 ms | 28.747 ms | neutral |

Outputs were bitwise exact.  HBM/CU contention consumed the nominal overlap;
the prototype was fully removed.

## Dequant grid saturation

At M27648, complete block-M64 routed latency was insensitive to expansion grid:

| blocks | balanced | skewed |
|---:|---:|---:|
| 416 | 30.633 ms | 28.066 ms |
| 832 | 30.693 ms | 28.202 ms |
| 1664 (service default) | 30.685 ms | 28.101 ms |
| 3328 | 30.601 ms | 28.015 ms |

Do not continue CTA-count tuning.

## Admission resweep

The same 73,724-token, 32-request heterogeneous code workload was rerun because
block-M64 changed the old admission cost curve.

| request split | five-round median input tok/s | decision |
|---|---:|---|
| 12+12+8 | 5744.65 | retain |
| 13+13+6 | 5635.06 | reject |
| 16+16 | 5518.93 | reject |

11+11+10 initially appeared slightly faster, but a seven-round repeat gave
5295.57/5416.29/5866.68/5873.45/5600.50/5623.65/5638.24, median 5623.65;
it is also rejected.  Do not infer admission wins from short noisy service runs.

## Next structural direction

The remaining viable CK direction is a gfx90a DSV4 B-loader which presents a
logical BF16 vector to the existing MFMA pipeline while reading raw/preshuffled
packed E2M1 plus E8M0 scale.  It must preserve the validated nibble/scale
contract and avoid the 3-GiB BF16 workspace.  The gfx950 `DeviceMoeGemmMX`
compute path is not reusable because it assumes native block-scaled MFMA.
