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

## Static HIP sorter and BS1/4/8/16 ABBA

The correctness-first PyTorch sorter was replaced by a single-block gfx90a HIP
histogram, Blelloch exclusive scan, and assignment scatter. It writes fixed
worst-case buffers (`M*T` groups and `4*M*T` assignment slots), never reads a
device occupancy scalar on the host, and is graph-safe. Invalid/remote EP
expert IDs require explicit handling: the first version filtered `-1` IDs but
left their dense output slots uninitialized. Real TP4+EP4 generation exposed
this immediately as gibberish even though all-valid unit tests passed. Invalid
assignments are now appended as parallel zero-fill work items, avoiding a
separate full-output `zero_` kernel. Unit tests cover both `-1` and out-of-range
IDs, and two real France prompts again returned exactly
`The capital of France is Paris.`

Formal HTTP ABBA used distinct prompts, a client start barrier, independent
HTTP sessions, one warmup plus five measured rounds per batch, 32 generated
tokens, temperature zero, no graph, and no speculative decoding. Results below
are aggregate completion tok/s; `trim` drops the fastest and slowest round.

| Arm | BS1 median/trim | BS4 median/trim | BS8 median/trim | BS16 median/trim |
|---|---:|---:|---:|---:|
| A1 indexed | 5.37 / 5.37 | 18.98 / 14.28 | 14.61 / 19.28 | 27.61 / 36.51 |
| B1 grouped | 5.21 / 5.20 | 18.30 / 13.83 | 9.30 / 17.57 | 17.70 / 30.85 |
| B2 grouped | 5.31 / 5.31 | 18.59 / 18.56 | 33.44 / 33.43 | 56.79 / 56.40 |
| A2 indexed | 5.25 / 5.25 | 18.75 / 18.74 | 33.26 / 33.61 | 58.14 / 58.12 |

The first pass was contaminated by extended first-service/JIT state, while the
stable B2/A2 comparison shows no grouped gain through BS16: BS4 and BS8 are
effectively tied, and grouped is about 3% slower at BS16. The old selector also
mistakenly used flattened down-projection rows (`M*topk`) as the batch size, so
BS1 down entered grouped. Gate/up and down now share one decision based on the
original token batch. The default grouped threshold is 32; lower thresholds
remain opt-in for future occupancy-bucket work rather than being claimed as a
performance win.

Output hashes were stable at BS1. At concurrent batches, both A1/A2 and B1/B2
showed similar greedy hash drift (for example 32/80 and 31/80 at BS16), so it is
baseline concurrent numerical nondeterminism rather than a sorter-specific
regression. Semantic correctness and completion length passed, but exact
concurrent bitwise parity remains an independent runtime issue.

## BS1 decode graph recovery

The initial Qwen correctness profile disabled decode graphs and sustained only
about 5.3 tok/s at BS1. Two independent capture blockers were fixed:

1. `DecodeCudaGraphRunner` called the now-zero-argument
   `require_{mlp,attn}_tp_gather` helpers with stale `server_args` arguments.
2. The HIP QSA fallback copied packed-KV offsets to the CPU and looped over
   requests. A BS1 fixed-extent implementation now masks the static Top-K
   buffer entirely on device. It is bitwise identical to the old fallback for
   valid KV counts 1, 17, and 64.

TP4/EP4 MQ4G128 then captured a full decode graph at BS1. Two independent
France requests returned exactly `The capital of France is Paris.` A 128-token
native-AR HTTP probe (six steady rounds) measured 22.18 tok/s median and 22.18
tok/s trimmed mean, about 4.2x the no-graph endpoint.

A 24-step GPU trace showed that the next dominant local-compute budget is the
MQ4 routed path: approximately 281 us/layer for gate/up and 431 us/layer for
down at graph shapes `[27,10]` and `[270,1]`, respectively. A row-persistent
assignment-scan prototype preserved France correctness but remained about
22.3 tok/s hot, so it was removed. Cross-rank reduce kernels show large
rank-dependent wait time in the trace and must be analyzed as graph critical
path rather than summed kernel duration.
