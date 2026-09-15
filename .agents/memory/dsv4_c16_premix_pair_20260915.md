# Pending independent-output-column pre-mix screen

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
