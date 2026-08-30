# DSV4 TP4 M32 production inverse-RoPE / wo_a split

Date: 2026-08-30

## Result

Four-rank rank-max graph timing on the real layer-20 M32 dump, using the
production HIP inverse-RoPE call and the actual production `wo_a` selector:

| profile | trimmed rank-max us |
|---|---:|
| inverse-RoPE + production wo_a | 41.431 |
| production wo_a | 39.449 |
| empty graph floor | 1.020 |
| force raw wave64 grouped kernel | 385.320 |

Therefore the isolated split is:

* production inverse-RoPE launch: **1.982 us/layer**;
* production M32 `wo_a`: **38.429 us/layer** above graph floor;
* forced wave64 grouped kernel: **384.300 us/layer** above floor.

Correctness passed 100 input mutations and 1000 graph replays for the
production profiles. The forced wave64 result is not bitwise equivalent to
the production einsum (`max_abs=0.015625`) because its wave reduction order is
different.

The inverse timing uses an identity complex frequency table. This makes the
in-place kernel replay idempotent and avoids inserting a copy into the graph;
it still launches the exact HIP production Triton kernel at the real M32xG2
tail shape.

## Important selector finding

At M32, production does **not** use the gfx90a wave64 grouped GEMV. The public
selector accepts only `1 <= M <= 8`; M16 and higher intentionally fall back to
batched einsum/GEMM so the weight matrix is reused across tokens. Thus the
earlier 76.467 us Torch-shaped knockout was only an optimistic gross bound,
not the production inverse+wo_a component time.

## Existing wave64 kernel resources and work decomposition

The instantiated kernel geometry is rows/wave=1, vector unroll=2, and four
wave64 waves per 256-thread workgroup. Each workgroup allocates an 8 KiB LDS
activation tile (`bf16 sx[4096]`), then each wave owns one output row and uses
two `float4` activation temporaries plus one FP32 accumulator. It uses vector
global loads, FP32 dot accumulation, and wave shuffle-down reduction; it does
not use MFMA.

For M32/G2/N1024 this creates:

```text
blocks/group = 1024 / (1 row * 4 waves) = 256
grid          = 32 * 2 * 256 = 16,384 workgroups
```

Every token independently scans both group weights. The weight traffic is
therefore approximately 32 times the reusable GEMM interpretation. This, not
minor VGPR occupancy, explains the forced kernel's 9.7x regression.

## Testable geometries and decision

The existing template permits rows/wave 1/2/4, unroll 1/2, and 4/8/16 waves.
Those knobs trade CTA count, accumulator pressure, and LDS/load amortization,
but none introduces cross-token weight reuse. They cannot plausibly close a
roughly 10x gap, so another geometry-only sweep is rejected.

A viable M32-specific experiment must change decomposition: one CTA should
own a weight-row tile and process multiple token rows (or call a CK/rocBLAS
batched GEMM), retaining FP32 accumulation and the accepted BF16 output. ISA
options such as `v_dot2_f32_bf16`/MFMA should only be considered inside that
weight-reuse decomposition; merely replacing the current dot primitive while
keeping token-owned CTAs will remain bandwidth-amplified.

Oracle: `scripts/rocm/bench_dsv4_tp4_m32_inverse_woa_split.py`.
Raw log: `/tmp/dsv4_tp4_m32_inverse_woa_split.log`.
