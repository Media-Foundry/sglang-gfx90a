# DSV4 C128 variable-N grouped projection oracle (2026-08-30)

## Scope

Standalone, service-free M32 oracle for the layer-21 C128 projection pair.  It
does not alter the production model path.  The strict reference is two BF16
`F.linear` calls:

- runtime `wqkv_a`: `[1536, 4096]`;
- compressor `wkv_gate`: `[1024, 4096]`.

The checkpoint-backed tensors are extracted by
`scripts/rocm/extract_dsv4_c128_projection_oracle.py`.  The script reconstructs
the runtime q/kv concatenation and block-128 FP8 dequantization order and uses
the existing real M32 attention-normalized activation only as a shape/input
trace.

## Reproducible tensor hashes

- wqkv BF16 SHA256:
  `4689df83023fc180bc67ef7a7967f80a5011f61d268d3f6bc15ed6828288328a`
- compressor core BF16 SHA256:
  `0f90e883622ff58c03d105347d5d74e9debdcec3d8a1db95f9c2bd3cd7906c17`
- M32 input SHA256:
  `e4a9e046a855bb9b88d0021bdd8bb72de4ac40adb42fe6bc4d1bdcc4a2ea8207`

## Results and stop decision

Two independent `F.linear` calls measured about `68.35--69.39 us`.  Concatenating
the weights into one N2560 call measured about `38.60 us`, so eliminating one
BLAS launch has an attractive approximately `30 us` upper bound.  It is not a
correct candidate: the N1536 q segment is bitwise exact, but the N1024 core
segment has `max_abs=1.220703125e-4`.

Padding both problems to N1536 and using `torch.bmm`/`torch._grouped_mm` also
measured about `38.53 us`, but the core segment remains non-bitwise with
`max_abs=1.52587890625e-5`.  A diagnostic N2048 core projection reproduces the
same values as the concatenated large-N algorithm, confirming that the mismatch
comes from the BLAS N-dependent algorithm/reduction association rather than
segment addressing.

`torch._grouped_mm` cannot represent this pair directly: its tensor dimensions
must be uniform and `offs` supplies variable M, not variable N.  The installed
hipBLASLt extension API does expose `hipblaslt_ext::GroupedGemm::setProblem`
with vectors of m/n/k, so true variable-N grouped GEMM is theoretically
expressible.  There is no existing PyTorch or repository binding, however;
using it would require a new C++ host wrapper with heuristic selection,
workspace and lifetime caching, stable pointer handling, and graph-safe replay.

### CK Tile variable-N follow-up

The local `/home/pc/Code/composable_kernel` fork has a maintained grouped-GEMM
ctypes bridge, so four legal gfx90a BF16 RCR `compv3/cshuffle/intrawave`
configurations were generated and run on the same two real groups.  Reported
kernel times (not host allocation/copy time) were:

- tile 32x64x32, warp tile 16x16x32: `70.33 us`;
- tile 32x128x32, warp tile 16x16x32: `96.45 us`;
- tile 64x64x32, warp tile 32x32x16: `68.13 us`;
- tile 64x128x32, warp tile 32x32x16: `95.31 us`.

All four fail both gates.  The best is slower than the `<48 us` continuation
threshold.  Against each projection's independent GPU `F.linear` reference,
the two tile-32 kernels have q/core `max_abs=0.00390625/0.0078125` with
`24360/16573` differing BF16 elements; tile-64 is similarly nonexact
(`24362/16568` differing elements).  This is the expected MFMA reduction-order
difference, not a grouped pointer/segment error.

Per the experiment's stop rule, do not hand-write a scalar/MFMA kernel or add a
production wrapper from this result alone.  The concat/padded number is only a
nonexact upper bound, and the available CK variable-N grouped path is both too
slow and nonexact.  A 100-mutation or seven-round ABBA run is therefore not
warranted.
