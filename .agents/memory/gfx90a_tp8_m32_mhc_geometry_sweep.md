# TP8 M32 MHC split-K geometry sweep

## Scope

- TP8, decode batch 32.
- Real layer-20 residual, function and RMS partial tensors.
- `_gfx90a_mhc_mix_splitk_stage0_kernel` followed by the production stage-1
  and tail kernels.
- Swept `BLOCK_N={1,2,4,8}`, `SPLITS={4,8,16}` and
  `BLOCK_K={512,1024,2048}` where geometrically valid.
- Seven-round CUDA-graph ABBA timings after correctness checks.

## Result

The production geometry (`BLOCK_N=4`, `SPLITS=8`, `BLOCK_K=1024`) measured
about 26.04 us.  The fastest valid candidate was
`BLOCK_N=4`, `SPLITS=16`, `BLOCK_K=1024` at 24.906 us, saving only
1.152 us/layer.  `BLOCK_N=4`, `SPLITS=8`, `BLOCK_K=2048` saved 1.018 us.

The BF16 output was exact for the leading candidates, while the intermediate
FP32 post/comb tensors differed at reduction-rounding scale (maximum absolute
differences about 8.94e-8 and 7.15e-7).  Combinations such as
`SPLITS=16`, `BLOCK_K=2048` are invalid because `BLOCK_K` exceeds the
1024-element split chunk and caused cross-split reads and large errors.

## Decision

Do not add a production selector or run a service A/B.  The best saving is far
below the 10 us/layer gate and some alternatives perturb FP32 reduction order.
The current M32 MHC geometry is already near its local optimum.  Continue with
attention producer-to-consumer fusion without introducing a shared completion
barrier between the compressor/indexer and main-attention branches.

Reproducer: `scripts/rocm/bench_dsv4_gfx90a_mhc_splitk_geometry.py`.
