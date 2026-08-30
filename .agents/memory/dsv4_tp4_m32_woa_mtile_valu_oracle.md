# DSV4 TP4 M32 wo_a cross-token VALU reuse oracle

Date: 2026-08-30

## Design

An independent HIP oracle changes the existing token-owned wave64 grouped
GEMV into `(token tile, group, N-row tile)` ownership. One aligned 16-byte
weight vector load is reused for MTile=2 or 4 token rows. Each token retains
the original K iteration, `gfx90a_dot8_f32`, and wave64 shuffle-down reduction
order. Output remains BF16. It consumes the original `[M,G,K]` and `[G,N,K]`
layouts and is not connected to production.

## Correctness

Across 100 real-dump-derived input mutations:

* MTile2 versus the original wave64 kernel: bitwise exact;
* MTile4 versus the original wave64 kernel: bitwise exact;
* either MTile versus production einsum: not bitwise exact, max abs 0.015625.

The production difference is expected: rocBLAS/einsum and the explicit wave64
shuffle tree use different FP32 reduction orders. The new decomposition itself
does not introduce any additional numerical change relative to the original
wave64 implementation.

## Seven-round rank-max ABBA results

| profile | trimmed rank-max us |
|---|---:|
| production einsum | 39.211 |
| original wave64 | 385.363 |
| MTile2 | 231.849 |
| MTile4 | 230.267 |

Both candidates fail the required `<33 us` / 15% threshold by a wide margin.
MTile4 is 82.97% slower than production. Do not connect it to production and
do not extend to MTile8.

## Code-object resources

Extracted directly from the compiled gfx90a HSACO metadata:

| geometry | LDS/workgroup | VGPR | SGPR | spills | wave/workgroup |
|---|---:|---:|---:|---:|---:|
| MTile2 | 16 KiB | 23 | 21 | 0 | 4 |
| MTile4 | 32 KiB | 25 | 23 | 0 | 4 |

Both use wave64 and a 256-thread workgroup. MTile4's 32 KiB LDS allocation
limits each CU to at most two resident workgroups from LDS alone; MTile2 can
fit four. Halving the grid from MTile2 to MTile4 therefore does not increase
effective latency hiding, explaining their nearly identical time. Neither
variant spills registers.

## Interpretation

Cross-token reuse does reduce the original wave64 runtime by about 40%, so the
weight-traffic diagnosis was directionally correct. It remains about 5.9x
slower than rocBLAS/einsum because the kernel still uses many independent
row-tile CTAs, stages full 4096-wide activations per CTA, and performs scalar
VALU dot accumulation rather than a matrix tile with broad weight reuse.

The next credible kernel experiment would be a CK/MFMA GEMM tile (M32 by an N
tile), not a larger MTile in this GEMV decomposition. MTile8 would consume the
full nominal 64 KiB LDS per workgroup, collapse occupancy further, and has no
credible route to the 33 us gate.

Files:

* `python/sglang/kernels/jit/csrc/gemm/gfx90a_bf16_grouped_gemv_mtile_oracle.cuh`
* `scripts/rocm/bench_dsv4_tp4_m32_woa_mtile_oracle.py`
* raw log `/tmp/dsv4_tp4_m32_woa_mtile_oracle.log`
