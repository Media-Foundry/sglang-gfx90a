# DSV4 TP4 M32 wo_a CK batched-GEMM screen

Date: 2026-08-30

The accepted TP4/M32 path computes two BF16 `M32 x N1024 x K4096`
projections for `wo_a`.  Production falls back to `torch.einsum` because the
gfx90a wave64 GEMV is intentionally limited to `M<=8`; the exact production
component oracle measures `38.429 us` for `wo_a`.

An isolated Composable Kernel build was configured under
`/home/pc/Code/composable_kernel/build-gfx90a-woa-rocm714c` with ROCm 7.14,
gfx90a and only the `batched_gemm` profiler.  The working compiler contract was
the same as the existing local CK build: CXX=`/opt/rocm/core-7.14/bin/hipcc`
and HIP compiler=`/opt/rocm/core-7.14/lib/llvm/bin/clang++`.  Two earlier
configure-only directories failed because base Conda's `_rocm_sdk_core` took
precedence or the wrong compiler path was supplied; neither compiled code.

After `amd-smi process` confirmed all GCDs idle, the profiler ran BF16 layout 1
with `G=2,M=32,N=1024,K=4096`, batch strides
`131072/4194304/32768`.  Thirteen gfx90a XDL instances were supported.  The
best was:

```text
84.624 us
DeviceBatchedGemmXdl<128,128,32,4,8,32,32,2,1>
```

Other candidates ranged from roughly 86 to 250 us.  The best CK instance is
therefore 2.20x slower than the current 38.429-us production einsum/GEMM, even
before charging any transpose/materialization needed to turn production
`[M,G,K]` into a batch-major contiguous input.  Do not connect generic CK
batched/grouped GEMM to production for this shape.

This result does not reject a purpose-built kernel that directly consumes the
original strided `[M,G,K]` layout and tiles multiple M rows per weight tile. It
does reject using the stock CK XDL instance set as that implementation.
