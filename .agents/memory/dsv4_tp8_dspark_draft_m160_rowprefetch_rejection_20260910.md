# TP8 DSpark draft M160 row-prefetch rejection (2026-09-10)

## Scope

- DeepSeek-V4-Flash original checkpoint weights.
- TP8 / EP1 / no A2A, 1,048,576-token pool.
- Strict DSpark gamma five with static uniform target verification: draft M160,
  target M192. No anchor-only or compact-target approximation.
- 32 fixed heterogeneous code requests, 1024 generated tokens per request.

## Component result

The existing subgroup-8 row-prefetch kernel was extended temporarily to M160.
On physical GCD 4 it passed 100 input/weight mutations and 1000 HIP Graph
replays bitwise exactly.

| routing | current full routed stage | row-prefetch | improvement |
|---|---:|---:|---:|
| balanced | 935.433 us | 874.271 us | 6.996% |
| skewed | 926.698 us | 840.184 us | 10.297% |

Logs: `/tmp/dsv4_tp8_m160_rowprefetch_balanced.log` and
`/tmp/dsv4_tp8_m160_rowprefetch_skewed.log`.

## Service result

Both control and candidate answered the C32 France sentinel exactly 32/32.
The candidate logged the M160 selector on every rank during graph capture.

| arm | warm resident C32 tok/s | mean accept length |
|---|---:|---:|
| control | 839.679 | 3.366 |
| candidate round 0 | 842.981 | 3.359 |
| candidate round 1 | 829.677 | 3.283 |

The candidate center is about 836 tok/s and has no measurable end-to-end gain.
The strict M192 target dominates the step, so a 7--10% improvement in only the
three-layer draft routed component is hidden by target verification and normal
run-to-run acceptance variation.

The first control round (771.496 tok/s) was excluded as a cold/service transient.
The candidate JSON was not finalized because the remaining redundant rounds
were interrupted after two complete printed records; their values are retained
in the session transcript. The completed control artifact is
`/tmp/dsv4_tp8_gamma5_m160_A_c32x1024.json`.

## Decision

Rejected for production and removed. Keep gamma-three draft M96 row-prefetch,
which has a measured service-level gain. Future gamma-five work must first make
the full target cheaper (or use a correctness-approved compact target); further
draft-only M160 tuning cannot move the 2k objective materially.
