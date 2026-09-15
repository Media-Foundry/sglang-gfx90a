# Independent-output-column pre-mix: component win, not integrated

The completed component results below supersede the initial pending notes.

Prepared while mixed-prefix ABBA owns TP8; **not GPU-run, no speed or exactness
result, no production selector change**. Files:
`.agents/experiments/dsv4_c16_premix_pair_20260915/{candidate.py,screen.py,README.md}`.
Only Python syntax checked. Ownership guard forbids GPU testing alongside service.

The latest accepted8K profile still has2.7037s/wave in two pre-mix boundaries.
Production pre-mix8 uses eight rows/one output column per CTA. This candidate
retains eight rows and computes two independent columns using the same loaded
activation vector. Each dot retains a separate shape[1,1024] reduction and16
K chunks, not a merged BLOCK_N=2 tensor. RMS consumption remains included.

Historical warning: `dsv4_tp8_m128_mhc_fusion_exactness_barrier_20260910.md`
records BLOCK_N2/4/8 changing floating reduction. The independent-tensor
formulation is an attempt to avoid that, not proof it succeeds; compiler
layout/contraction/register pressure must still be tested. A targeted memory
search found that negative history, not a prior validated large-M independent
column-pair result; do not interpret an incomplete search as proof none exists.

After TP8 releases GPUs, physicalGPU4 screen versus actual productionpre-mix8:
M17/128/8192/32767/32768, captured residual/Fn expanded/scaled for occupancy,
exact outputs, row permutation,1000 graph replays plus mutation, full component
ABBA and resource/HSACO evidence. Stop on any inexact result; only about10%
consistent large-shape gain merits stronger random-parameter/real-layer checks
and a subsequent service trial. No claim this optimization applies to the
legacy batch1 FP16-Fn fused-tail path or fixes global drift.

## Completed isolated screen

After mixed-prefix ABBA completed and all service owners stopped, physicalGPU4
only. `screen.json` uses captured/scaled real layer0 residual/Fn;10 mutations
per M, row permutation,1000 graph replays and changed-input replay all exact.
Its driver source is the version committed in90bd1a154a (before randomize flag).

| M | Production pre-mix8 ms | Independent-pair ms |
| ---: | ---: | ---: |
| 17 | .114787 | .140275 |
| 128 | .124163 | .145091 |
| 8192 | 1.886318 | 1.379889 |

`full.json`:100 random temporary Fn/activation mutations per shape, EPS cycling,
row permutation,1000 graph replays and changed-input replay, all FP32 output
bits exact. These operations do not alter checkpoint files. Full pre-mix/RMS
consumption measured with three eagerABBA cycles, five calls/event:

| M | Production pre-mix8 ms | Independent-pair ms |
| ---: | ---: | ---: |
| 32767 | 7.620457 | 4.993055 |
| 32768 | 7.639178 | 4.989183 |
| 65536 | 15.277621 | 9.658151 |

AtM32768 latency falls34.69% (operator rate +53.11%); these are not E2E gains.
Registers51->78 (at65536:52->79), reported spills0/LDS0; HSACO hashes retained.
Both GPU scripts exited0. Small shapes lose and must stay outside the selector.

Next: default-off integration only under existing original-V4 TP8 ordinary
prefill mix-reuse context, K1024, group8 and8192<=M<=65536. Require actual
selector-hit oracle and scope checks, then C16x8K E2EABBA. Do not enable on
batch1 FP16 fused-tail, AR, DSpark, TP4 or V4.1. Do not add this component
gain to the mixed-prefix21.35% service result; they are different paths/tests.
