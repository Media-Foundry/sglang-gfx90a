# Qwen3.8-Next routed-expert MQ4G128 audit

Date: 2026-08-27

## Format identity

HipFire/HipScope `MQ4` means MagnumQuant, not MXFP4. MQ4G128 applies a
group-local FWHT-128 transform and stores each 128-weight group as 72 bytes:
an FP32 scale, FP32 zero point, and 64 packed-nibble bytes. The activation is
rotated with the matching FWHT before the matrix product. The underlying
HFQ4G128 projection kernels are rotation-agnostic.

## Why G128 fits Qwen4Exp

Qwen4Exp routed experts use H=2560, I=640 and top-k=10. Both contraction
dimensions divide exactly by 128:

```text
gate/up K = 2560 = 20 * 128
down    K =  640 =  5 * 128
```

G256 would require padding down K from 640 to 768. G128 consumes 4.5 bits per
weight versus G256's 4.25 bits, so the overall three-projection routed expert
footprints are close:

```text
G128, no padding:       56.25% of one-byte/weight storage
G256, padded down K:    56.67% of one-byte/weight storage
```

G128's material advantage is avoiding 20% wasted down-projection arithmetic,
not a large additional memory saving. Applied to the checkpoint's 112.514 GiB
of routed experts, G128 is approximately 63.3 GiB before container metadata,
freeing roughly 12.3 GiB/GCD under TP4+EP4 compared with the current FP8
checkpoint.

## Reusable HipScope implementation

`/home/pc/Code/hipscope` already contains:

- `quantize_hfq4g128`
- MQ4G128 FWHT-128 activation rotation
- G128 indexed MoE gate/up
- atomic-free G128 indexed MoE down plus separate deterministic combine

The indexed kernels are named `gemv_paro_q4g128_*`, but their projection math
only assumes the common 72-byte HFQ4G128 layout; MQ4G128 can reuse them when
the caller supplies FWHT-128-rotated activations. Existing code uses a
32-thread block and therefore occupies only half a native gfx90a wave64. A
wave64 two-row port analogous to HipScope's G256 CDNA kernel is the first
obvious optimization.

## gfx90a true-shape microbenchmark

The probe used H=2560, I=640, top-k=10 and four resident synthetic experts on
one otherwise idle MI250 GCD. It timed only indexed gate/up and down projection;
it excluded FWHT, SwiGLU, top-k sorting and final down combine. Every output was
finite and nonzero. This was deliberately a conservative all-ten-experts-on-one
GCD stress case; EP4 normally assigns fewer local expert rows.

| Batch | gate/up (us) | down (us) | total (us/layer) |
|---:|---:|---:|---:|
| 1 | 74.98 | 36.93 | 111.91 |
| 4 | 277.74 | 139.47 | 417.21 |
| 8 | 547.39 | 276.00 | 823.39 |
| 16 | 1088.52 | 550.39 | 1638.91 |

At BS1, approximately 27.65 MB of selected expert weights / 111.91 us is about
247 GB/s from the half-wave scalar kernel, leaving credible room for a wave64
two-row implementation. Scaling is almost exactly linear with batch because
the indexed GEMV rereads weights for every token. It is promising for TP4
low-concurrency decode and context-capacity recovery, but not a high-concurrency
endpoint until expert occupancy sorting and a grouped G128 MMQ/MFMA kernel
reuse weights across token assignments.

## Integration gates

1. Quantize routed experts only; keep attention, GDN, PLE, shared experts,
   embeddings and LM head in checkpoint FP8/BF16.
2. Preserve the current FP8 TP4+EP4 model as the correctness oracle.
3. Implement per-expert G128 packed storage and FWHT-128 rotation at gate/up
   and down inputs.
4. Validate one selected expert projection against dequantized FP32 first,
   then teacher-forced per-layer/logit parity, then short semantic generation.
5. Measure quantization quality before treating memory fit as success. Plain
   affine 4-bit conversion may need AWQ/GPTQ calibration.
6. Use indexed wave64 for BS1/sparse occupancy and grouped G128 MMQ for higher
   occupancy; do not use the current per-token GEMV as the BS8/BS16 endpoint.

## SGLang integration checkpoint

Implemented on 2026-08-27 as an opt-in routed-expert-only method selected by:

```text
SGLANG_QWEN4_GFX90A_MQ4G128_ROUTED=1
```

Selection fails closed unless the FusedMoE shape is exactly H=2560, I=640,
top-k=10 on HIP. The original block-FP8 loader remains responsible for reading
the checkpoint. Post-load processing streams four experts at a time through
FP8 dequantization, FWHT-128 and affine G128 packing, then rebinds the existing
Parameter's `.data` so loader-held Parameter references do not retain the old
FP8 storage.

Two execution tiers are present:

- wave64 indexed: two output rows per physical wave, used for low occupancy;
- expert-sorted A4 grouped: loads each weight group once for up to four token
  assignments, selected when mean live assignments/expert reaches the explicit
  occupancy threshold.

Correctness gates passed on gfx90a:

1. indexed and grouped kernels independently match a dequantized FP32 matrix
   oracle (`rtol=atol=2e-5`);
2. streamed FP8-to-MQ4G128 conversion is bitwise identical to whole-tensor
   conversion;
3. Qwen true-shape gate/up -> SwiGLU -> down -> router-weighted sum matches the
   dequantized FP32 oracle;
4. a real TP4+EP4 service returned exactly "The capital of France is Paris."
   twice, 8 completion tokens and normal stop each time.

Real TP4+EP4 model memory changed from 43.80 GiB/GCD (FP8) to 34.38 GiB/GCD
(MQ4G128), saving 9.42 GiB/GCD. At `mem_fraction_static=0.65`, the server still
allocated 276,096 BF16 KV tokens and retained about 22.2 GiB/GCD after memory
pool setup. Initial whole-layer FP32 conversion OOMed at ~62 GiB; four-expert
streaming removed that peak. Replacing the Parameter object also temporarily
retained both FP8 and packed weights (59.75 GiB); preserving Parameter identity
via `.data` rebinding fixed it.

The current A4 sorter is a correctness-first PyTorch implementation and reads
occupancy to the host. It is not CUDA-Graph-safe and is not the final BS8/BS16
path. Replace it with a static-buffer HIP histogram/scan sorter before claiming
high-concurrency performance.
