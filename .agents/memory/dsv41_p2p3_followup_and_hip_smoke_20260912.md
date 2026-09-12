# V4.1 P2/P3 follow-up and HIP smoke (2026-09-12)

## P2/P3 result

The storage audit is a post-load live-storage census; it does not add logical
tensor bytes, allocator reserved bytes, and driver-used bytes.  On the clean
TP8/AIter/host-Engram start (`--max-total-tokens 8192`) every rank reported:

```text
named CPU storage:  logical=23.604 GiB, unique=23.604 GiB (2 mmap storages)
named GPU storage:  logical=68.208 GiB, unique=53.958 GiB
GPU allocator:      allocated=54.156 GiB, reserved=62.211 GiB, peak=56.117 GiB
routed weights:     42.188 GiB (float4_e2m1fn_x2)
routed scales:       2.930 GiB (float8_e8m0fnu)
direct W2 clones:    0 GiB; GPU Engram: 0 GiB
host mmap:          23.604 GiB across two tables
after empty_cache:  allocated=54.156 GiB, reserved=59.414 GiB
```

The 14.25-GiB logical/unique difference is aliases/views (eight shared
storage groups), not eight independent copies.  The post-`empty_cache`
allocator delta is retained inactive blocks, not live model parameters.  The
unregistered live gap was about 0.197 GiB and the driver-side allocation was
about 0.9--1.0 GiB.  These figures do not prove that two different storage
pointers contain identical bytes; a content hash would be required for that.

For the active AIter path, `Fp8MoEMethod` creates routed E8M0 scales with
`torch.float8_e8m0fnu`; the legacy shuffle and the 288-to-384 padding retain
one-byte scales.  The measured 5.93 GiB/GCD number is only the extra cost of
the non-AIter fallback's local routed scales becoming FP32:

```text
unpadded E8M0 routed scales: 1.9775 GiB/GCD
fallback FP32 scales:        7.9100 GiB/GCD
delta:                       (4 - 1) * 1.9775 = 5.93 GiB/GCD
```

It is not a full-checkpoint scale expansion.  Engram's E8M0 scales stay in
the host mmap and are not included in GPU VRAM.  The principal live AIter
overhead is geometry/layout padding: about 11.206 GiB/GCD for 288 to 384,
plus about 0.293 GiB/GCD for the legacy W2 scale-axis pad, or about 11.499
GiB/GCD versus an unpadded E8M0 baseline.

The fallback A/B was a fresh process with host Engram enabled but AIter
omitted.  It reported `routed_weights=31.641 GiB` and
`routed_scales=7.910 GiB` (FP32), while the AIter run reported
`42.188/2.930 GiB`.  An intentionally GPU-resident Engram run failed while
allocating a 270-MiB block with only 180--244 MiB free, so the host-table
default is required for this 64-GiB/GCD configuration.  The failed run did
not reach a post-load census and is not used as a duplicate-storage count.

## Correctness/backend smoke caveat

The host-backed TP8 process loaded successfully with the `triton` attention
backend and served a request, but the returned text was visibly corrupted.
Earlier runs with the slow host CPU-reference Engram lookup and with Engram
disabled also returned corrupted text, so this observation does not isolate
the host mmap as the cause.  The `unified_kv_triton` backend currently stops
at an assertion because V4.1 has ratio-1/2 pools; the default TileLang path
failed HIP module initialization.  Therefore the storage conclusion is
independent of, and must not be presented as, a V4.1 correctness result.

## HIP FP8 JIT fix validated locally

ROCm JIT compilation previously failed because the generic 8-bit quantizer
unconditionally included `cuda_fp8.h`.  The HIP branch now uses
`hip/hip_fp8.h` and `__hip_cvt_float2_to_fp8x2` with the explicit E4M3
interpretation.  Tiny random tests for row-major and packed column-major
layouts, group sizes 16/32/64/128, fused and unfused modes, matched PyTorch
FP8 bytes and scales on gfx90a.  The row-major dispatch also now forwards the
runtime `fuse_silu_and_mul` flag; previously it silently instantiated the
unfused template.  This fix is separate from the P2/P3 accounting and does
not claim V4.1 end-to-end correctness.

## Follow-up startup validation (2026-09-13)

With an intentionally unregistered private host mmap (`PIN=0`), the service
loaded all 48 shards and initialized the V4.1 pools after raising
`mem_fraction_static` to 0.96. The first attempt at 0.80 was rejected because
the loaded model left no KV-cache headroom; this is a capacity setting error.
The launcher now disables decode CUDA-graph capture for this fallback because
dynamic GPU-index to CPU copies cannot be captured from pageable memory.

The long-lived foreground retry reached model initialization without the
previous graph-capture exception or an HSA memory fault, but then remained in
CPU scheduler initialization and never opened its HTTP port. It was terminated
after the process was confirmed to be a startup busy-wait. The next correctness
step is to use the normal registered/pinned host path and separately diagnose
that scheduler initialization wait; no claim of end-to-end V4.1 correctness is
made from this run.
