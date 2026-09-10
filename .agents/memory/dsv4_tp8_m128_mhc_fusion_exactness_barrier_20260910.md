# TP8 M128 MHC fusion exactness barrier (2026-09-10)

## Scope

Component oracles only, single GCD 0, real layer-20/rank-0 tensors from
`/tmp/dsv4_tp8_rowstable_router_all_20260908` (leading 128 of 736 real rows, no
tiling). No production selector, env default, or service was changed. An
unrelated TP8 diagnostic service held all eight GCDs; only free VRAM was used.

## Why the C32 path misses several fused kernels

`global_batch_size` is `forward_batch.batch_size`, so strict C32 DSpark passes
32 and every `global_batch_size == 1` gate in `mhc.py` fails, even though the
target verify is M128 rows. Four gates are involved: splitk pre-mix, splitk
fused tail, native post-pre, and the fused weighted-RMS tail. The splitk pre-mix
gate carries a stated reason (192-CTA geometry must avoid the Mori progress
kernel); the strict TP8 profile is no-A2A, so that reason does not apply there.

## Segment sizes at M128 (independent medians, not a critical path)

| segment | C32 actual | gbs=1 fast path | delta |
|---|---:|---:|---:|
| post-combine (rms variant) | 23.638 us | -- | -- |
| pre-mix from_partials | 74.992 us | splitk 55.746 us | 19.246 us |
| sinkhorn (triton, 20 iters) | 8.342 us | -- | -- |
| tail weighted-sum + RMS | 12.741 us | fused 8.520 us | 4.221 us |

Pre-mix is about two thirds of the boundary and dominates the pool; the tail is
the smallest term. The segment sum (139 us) exceeds the layer-20 marker's
116.88 us MHC span, so these must not be added as an exact critical path.

## The barrier: every faster geometry changes the numerics

Measured against each path's own production baseline, over real inputs:

| candidate | speed | vs production |
|---|---:|---|
| fused weighted-RMS tail (BLOCK_H=4096) | 8.52 us vs 12.74 | 1 bf16 ULP, `max_abs=0.015625` |
| splitk pre-mix (K split 8) | 55.75 us vs 74.99 | `mixes max_abs=2.67e-05`, 2078/3072 elems |
| pre-mix BLOCK_N=2/4/8 | 57.4--61.6 us vs 74.9 | `max_abs=3.8e-06--1.14e-05` |

An isolation diagnostic separated the tail's cause: the accumulation expression
form (4-term explicit vs `static_range` loop) is irrelevant -- both are exact at
`BLOCK_H=256` and both differ at `BLOCK_H=4096`. Tile width alone decides it,
via compiler FMA contraction. For the pre-mix, `tl.sum` reduces along K, so its
reduction tree changes with tile shape; BLOCK_N was expected to be exact and was
not.

A chunked fusion attempt (256-wide weighted sum for contraction, then one
full-row RMS) is invalid, not merely inexact: storing to the output in one loop
and reading the whole row back in a later loop of the same program has no
inter-warp barrier. It measured `max_abs=8.85` at `num_warps=4`. Keeping the row
in registers instead would need 16 KiB of VGPR per program and would restore the
4096-wide reduction. Tail fusion and bitwise equality are therefore structurally
exclusive: the production contraction is fixed by 256-wide tiles, while any
in-register RMS must see the full row.

## Decision

Do not wire any of these into the C32 path on component speed alone. Two
precedents govern: `dsv4_tp4_prefill_mfma_drift_rejection_20260906` rejected a
+4.8% non-bitwise prefill change because top-1 IDs moved semantically and noted
a France smoke test is too weak to accept such a change; and
`dsv4_tp8_dspark_long_decode_m128_followup_20260910` records the current C32
baseline as `cross_round_all_exact=false`. While the baseline itself is not
reproducible across rounds, a drift test cannot adjudicate a non-bitwise
candidate -- the control drifts too. Restoring a reproducible baseline is a
prerequisite for spending the MHC pool, not an afterthought.

Reproducers:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python python \
  scripts/rocm/bench_dsv4_m128_mhc_weighted_rms_fusion_oracle.py \
  --mutations 100 --graph-replays 1000 --rounds 7
```

Diagnostics kept at `/tmp/mhc_fusion_exactness_diag.py`,
`/tmp/mhc_boundary_segments.py`, `/tmp/mhc_splitk_exactness.py`,
`/tmp/mhc_premix_blockn.py`, `/tmp/mhc_tail_exact_fusion.py`.

## One exact candidate found, and the rule that predicts exactness

Fusing the four HC outputs of `mhc_post_combine_rms_triton` into one CTA
(grid `(M,4,16)`=8192 -> `(M,16)`=2048, reading `x` and the four residual
channels once instead of four times) is bitwise exact on both outputs over 100
real-input mutations, at `num_warps=4` only:

| variant | out ne | partials ne | time |
|---|---:|---:|---:|
| fused4 `nw=1` | 49 | 2498 | 14.621 us |
| fused4 `nw=2` | 57 | 2262 | 16.240 us |
| **fused4 `nw=4`** | **0** | **0** | **20.144 us** |
| production | -- | -- | 23.659 us |

`nw=4` is production's warp count: 4x64 lanes maps exactly onto the 256-wide
tile. Fewer warps redistribute elements per lane and change contraction.

The predictive rule from all experiments: **exactness follows the (tile width,
warp count) pair, not the expression form.** Fusing *outputs* with
`static_range` while holding the tile fixed is exact; changing *tile geometry*
never is. Four pre-mix variants confirm the negative half -- splitk, `BLOCK_N`
2/4/8, an N-loop with 1D `[BLOCK_K]` tiles, and an N-loop keeping the literal
2D `[1,BLOCK_K]` tile -- are all inexact (`3.8e-06` to `2.7e-05`). A 1D
`tl.sum` over `BLOCK_K` does not reproduce a 2D `tl.sum` over `[1,BLOCK_K]`
even at identical logical width.

## Pool size: too small to be the 2k main line

Against the 95.8250 ms measured C32 step, 43 layers x 2 boundaries:

| candidate | us/layer | % of step | exact |
|---|---:|---:|---|
| post-combine fused4 | 7.030 | 0.315% | yes |
| tail fusion | 8.442 | 0.379% | no |
| pre-mix splitk | 38.492 | 1.727% | no |
| all three stacked | 53.960 | 2.422% | -- |

The exact-only win is 7.03 us/layer, below the established 10 us/layer gate, so
`fused4` is **not** wired into production on this evidence; it is kept as a
verified component result. The claim that the two MHC boundaries are a ~233 us
pool is right as a span but wrong as an opportunity: the addressable part is
~54 us/layer, and only 7 us of that is exact.

## What this makes the actual blocker

47 us/layer (pre-mix splitk + tail) sits in the useful range and is locked
behind reproducibility, not behind kernel work. Since
`dsv4_tp8_dspark_long_decode_m128_followup_20260910` records the C32 control as
`cross_round_all_exact=false`, a token-drift test cannot distinguish a
1-ULP candidate from control noise. Restoring a reproducible strict C32 DSpark
baseline is therefore the prerequisite for spending this pool, and native AR at
TP8 is already fully reproducible (`dsv4_tp8_drift_fix_validation_20260907`,
six gates), so the fault is specific to the speculative path.

Diagnostics: `/tmp/mhc_post_combine_fused4.py`, `/tmp/mhc_premix_nloop.py`,
`/tmp/mhc_premix_nloop2d.py`.
