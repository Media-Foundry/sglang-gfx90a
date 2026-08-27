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

## TP4/EP1 non-aligned expert-shard experiment

A real TP4/EP1/no-A2A MQ4G128 path was brought up experimentally for the
`640 / 4 = 160` local expert intermediate. The generic block-FP8 loader cannot
shard the checkpoint's five 128-row scale blocks correctly: ranks need
overlapping source blocks `[0,1]`, `[1,2]`, `[2,3]`, and `[3,4]`. A temporary
MQ4 loader retained the canonical local 160-row FP8 slice, loaded those two
overlapping scale blocks, reconstructed scales with each rank's global
`0/32/64/96` block offset, and padded the down input from 160 to 256 for G128.
On the real layer-0/expert-0 checkpoint tensors, all four ranks' packed w13 and
w2 bytes exactly matched full-FP8-dequantize, global-slice, then MQ4-quantize.

The full service loaded at about 43.64 GiB/GCD, captured graph BS1, and returned
`The capital of France is Paris.` exactly in 10/10 requests. Three hot
128-token native-AR requests measured `22.73 / 22.64 / 22.64 tok/s`, median
`22.64 tok/s`, versus the TP4/EP4 checkpoint's `22.18 tok/s`: only about 2.1%.
The small expert-load-balance gain is largely offset by the 160-to-256 down
padding, all ten expert shards executing on every rank, and TP reduction. One
long greedy prompt also produced two completion hashes across three hot runs,
so this branch did not meet the performance or strict-parity retention gate.
The production changes were removed; TP4/EP4 remains the validated default.

## gfx90a hyperconnection mix/combine checkpoint

The Qwen4 decode graph used a persistent Triton HC mix grid with all 110 gfx90a
CUs, eight warps per CTA, and `BLOCK_R=64`. On the production shape
`rows=1, K=10240, lowrank=320, HS=2560`, a geometry sweep reduced the median
kernel time from about 98.7 us to about 54.0 us with 100 CTAs, four warps, and
`BLOCK_R=32`; sampled outputs were BF16 bitwise identical. This geometry is
selected only on gfx90a, while other architectures retain the original grid.

The first HIP guard relaxed in checkpoint `1398d7f714` belongs to the fused
`GroupedGemmaRMSNorm`, not HC combine. A post-commit decode trace exposed this
attribution error: the trace contained the grouped RMS kernel but still showed
the torch.compile combine GEMM/elementwise chain. The performance result is
valid, but its corrected ABBA attribution is:

- A1, tuned mix and old HIP grouped RMSNorm: `22.99 tok/s`;
- B1, tuned mix and fused gfx90a grouped RMSNorm: `24.287 tok/s`;
- B2, independent service: `24.367 tok/s`;
- A2, independent old grouped-RMS service: `22.937 tok/s`.

Thus the gfx90a grouped RMSNorm path contributes about 6.1% on top of the tuned
mix geometry.

HC combine remained a second omission. Its existing fused C++ kernel was also
guarded by `not _is_hip`, although the implementation is portable through the
SGLang device abstractions. On gfx90a the fused kernel measured about 15.7 us
for the production shape versus about 145.1 us for the torch.compile chain,
with bitwise equality to the eager BF16 reference. Relaxing this separate guard
only for gfx90a produced a second service-level ABBA:

- A1, grouped RMSNorm on and old HIP combine: `24.367 tok/s`;
- B1, fused gfx90a combine: `25.225 tok/s`;
- B2, independent service: `25.290 tok/s`;
- A2, independent old-combine service: `24.381 tok/s`.

The real fused-combine gain is therefore about 3.6%. The final tuned-mix,
grouped-RMS, and fused-combine stack is about 13.1% faster than the earlier
PLE-on 22.36 tok/s baseline. Every service returned the exact France sentence;
both final B services passed the correctness probe, and the dedicated HC
mix/combine tests passed 25/25 on gfx90a. The combined kernel suite passed
53/54; every production `H=10240, group=2560` grouped-RMS case passed, while
one unrelated BF16 `M=1024, H=2048, group=1024` case exceeded the existing
tolerance at 1 of 2,097,152 elements (`0.015625` absolute difference).

## Single-kernel gfx90a FWHT128

The MQ4 activation rotation was a hidden launch-count bottleneck. Each
`fwht128()` used seven eager `torch.stack((left+right, left-right))` stages.
Since every routed layer rotates once before gate/up and once before down, the
old graph launched about 14 cat/copy kernels and 28 add/sub kernels per layer.
The final decode trace's cat and add counts matched this construction.

A gfx90a Triton kernel now keeps all 128 values in one program and performs the
seven XOR-partner butterfly stages with `tl.gather`. It emits one graph-safe
kernel per rotation and retains the eager implementation elsewhere. Across
production shapes `[1,2560]`, `[10,640]`, `[32,2560]`, and `[320,640]`, output
was bitwise identical to the old FP32 implementation. Representative micro
times were `22.3 us` versus `262.3 us` for `[1,2560]`, and `22.5 us` versus
`267.8 us` for `[10,640]`. The MQ4 gate/down oracle passed 3/3.

TP4/EP4, no-A2A, MQ4G128, PLE-on, graph-BS1 128-token native-AR ABBA was:

- A1, eager seven-stage FWHT: `25.290 tok/s`;
- B1, fused FWHT: `30.470 tok/s`;
- B2, independent fused service: `30.495 tok/s`;
- A2, independent eager service: `25.288 tok/s`.

The fused FWHT therefore improves single-request decode by about 20.5%. Both
fused services passed the France correctness probe; B1 passed 10/10 exact
France responses. Greedy long-output hash drift remained present in both A and
B and is not introduced by the bitwise-exact FWHT kernel.

## TP4/EP1 retest after FWHT and HC optimization

The earlier non-aligned TP shard loader was restored temporarily on top of the
final FWHT/HC stack. It loaded the overlapping FP8 scale blocks for each
160-row TP shard and zero-padded the down-projection input to 256 for G128. A
new generic quantized-MoE compatibility check also required a narrowly scoped
exception for this explicit padded path. The service captured graph BS1 and
returned `The capital of France is **Paris**.` with the same 9-token sequence
and normal stop in 10/10 requests.

With identical PLE and GDN/QK alternate-stream switches, TP4/EP4 measured
`31.390 tok/s` trimmed over seven 128-token requests. TP4/EP1 hot requests were
`31.634--31.706 tok/s` (about `31.67 tok/s` representative), only about 0.9%
faster. EP1 used 43.64 GiB/GCD for weights and required
`mem_fraction_static >= 0.695`, versus the materially smaller EP4 footprint.
Long greedy output hashes still drifted, as they did in the EP4 arm. The gain
is below the 5% retention threshold and does not justify the padding and memory
cost; the TP4/EP1 loader and compatibility exception were removed again.

## Router-weighted down-projection experiments

Two variants were rejected. A fully fused kernel that serially evaluated all
ten selected experts inside one wave reduced the isolated chain from about
72.1 us to 43.7 us, but destroyed assignment parallelism and reduced the full
service to 9.39 tok/s. A second variant retained the existing
`(N/2, assignment)` grid and only folded the router scalar multiply into each
down-projection output. The production down shape improved from 60.64 us to
50.08 us in isolation and its quantized-weight oracle passed, but the TP4/EP4
service measured only 34.428--34.539 tok/s (trimmed 34.481) versus the committed
34.329 tok/s checkpoint, about +0.44%. France was exact in 10/10 requests. The
gain is below service noise and the 5% retention threshold, so both variants
were removed. This confirms that the separate elementwise router multiply is
not a material graph-critical-path bottleneck; future down fusion must also
remove a larger producer/consumer boundary without reducing expert-slot
parallelism.

Two compute-format probes were also rejected. A packed G128 kernel converted
affine nibbles and activations to FP16 pairs and used CDNA2
`v_dot2_f32_f16` with FP32 accumulation. With pre-converted FP16 activations,
gate/up improved only `56.64 -> 54.56 us` while down regressed
`37.44 -> 37.76 us`; including the activation cast made both paths slower.
Relative L2 error was about `6.5e-4--7.3e-4`. A full FP16 expert-shadow probe
removed nibble decoding entirely but expanded weight traffic by 4x. The first
8-row/block geometry changed gate/down from `52.96/37.28 us` to
`73.44/52.48 us`; matching the dense FP16 kernel's 16-row geometry still gave
`55.52 -> 68.48 us` for gate/up. Thus scalar FP32 FMA issue rate is not the
dominant packed-kernel cost, and spending memory bandwidth on FP16 shadows is
counterproductive on this BS1 route. Both prototypes were removed.

## Fused routed SwiGLU/FWHT and gfx90a QSA packed decode

Two independent decode consumers were fused on the TP4/EP4 MQ4 graph. First,
the routed gate/up epilogue now feeds one Triton kernel that computes FP32
SwiGLU and the following five FWHT128 groups without materializing the
intermediate `[M,T,640]` tensor. The redundant BF16-to-FP32 cast before the
first FWHT was also removed. Standalone `[1,10,1280]` latency changed from
about 100 us for the old pointwise-plus-FWHT pair to 52.5 us; maximum FP32
difference was `1.79e-7`. Service hot decode increased from 31.39 to about
31.85 tok/s, roughly 1.5% alone.

Second, the HIP BS1 QSA fallback previously converted the fixed 2048-row
packed K/V scratch to FP32, materialized scores and probabilities, and launched
two einsums plus softmax. A gfx90a HIP wave64 kernel now handles the validated
local shape `q=[1,6,256]`, `K/V=[2048,1,256]`. Eight dynamic KV splits produce
FP32 `(max,sum,numerator)` partials with 48 CTAs; six reduction CTAs merge them
and emit BF16 output. This keeps all valid KV lengths on the fast path rather
than optimizing only short contexts. Representative micro latency versus the
old roughly 381--384 us fallback was:

| valid KV | old PyTorch | gfx90a split-K |
|---:|---:|---:|
| 193 | 383.5 us | 44.8 us |
| 512 | 382.7 us | 55.7 us |
| 1024 | 381.0 us | 73.4 us |
| 2048 | 381.1 us | 113.1 us |

The QSA oracle covered valid lengths `1/64/193/512/1024/2048`; maximum error
was `1.22e-4` before BF16 output comparison, with several lengths bitwise
equal. The combined MQ4/QSA suite passed 12/12. Decode graph BS1 captured and
France returned exactly `The capital of France is **Paris**.` in 10/10 calls.

Formal same-code ABBA used identical PLE, MQ4 and GDN/QK alternate-stream
settings, seven 128-token native-AR requests per main arm, and explicit switches
for both new consumers:

- A1 old consumers: trimmed `31.390 tok/s`;
- B1 fused SwiGLU/FWHT plus single-CTA QSA: trimmed `34.141 tok/s`;
- B2 fused SwiGLU/FWHT plus 8-way QSA: trimmed `34.329 tok/s`;
- A2 old consumers: trimmed `31.431 tok/s`.

B2 is about 9.2% above the A1/A2 mean. Long greedy hashes continue to drift in
both old and new arms; this known baseline issue is not introduced by the QSA
kernel, while the fixed France oracle remains exact.
