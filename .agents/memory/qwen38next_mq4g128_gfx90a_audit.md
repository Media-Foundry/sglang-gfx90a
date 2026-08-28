# Qwen3.8-Next routed-expert MQ4G128 audit

## 2026-08-28: post-49.7 tok/s decode probes rejected

All service experiments below used native AR, TP4/EP4/no-A2A, decode graph
BS1, and scanned `amd-smi process` before GPU work.  The restored control arm
with routed MQ4G128 and Qwen-wide wave64 BF16 linears produced nine steady
256-token rounds at `49.36--49.72 tok/s` (median about `49.57 tok/s`), all with
one completion hash `815dba1b46a50050`.

- An indexed MQ4 kernel computed two adjacent output rows per wave32 subgroup
  and shared activation loads.  True-shape gate/up ABBA improved
  `40.62 -> 34.35 us` (15.4%), while the short-K down projection regressed, so
  the selector was restricted to gate/up.  Full-service B nevertheless measured
  only about `49.65 tok/s` median versus `49.57 tok/s` A (about 0.16%).  B was
  internally hash-stable but followed a different greedy trajectory because
  interleaving two FP32 accumulators changes compiler FMA scheduling.  A
  four-row variant reached about `32.75 us` once but did not survive repeated
  timing.  The pair/quad kernels, selector, and environment flag were removed.
- Qwen HC down rows-per-wave 2 -> 1 passed the dedicated oracle, but the whole
  two-stage HC micro changed only `31.51 -> 31.08 us` (1.37%).  It was reverted
  without a service run.
- Reusing routed MQ4G128 for a dense `4096x2560` projection was slower than the
  committed BF16 wave64 scan: `35.79 -> 48.79 us`.  Packed-weight bandwidth did
  not repay affine nibble decode, so no dense MQ4 path was retained.
- The existing per-row INT8 + dynamic activation `sdot4` GEMV was first extended
  to Qwen shapes.  Its original form redundantly quantized the activation in
  every output block.  A second prototype quantized once into a static device
  workspace and launched a prequantized `sdot4` consumer.  It still regressed
  `4096x2560` from `35.78 -> 48.08 us` and `1536x2560` from
  `30.13 -> 44.50 us`, with roughly 1.2% relative L2 error.  Both variants and
  their relaxed shape guards were removed.
- A clean routed-MQ4 graph trace proved that the default-on Qwen Top-10 switch
  does not select its single-wave kernel unless import-time `SGLANG_USE_AITER`
  is enabled; the 49.6 tok/s service therefore still used AOT
  `topkGatingSoftmax` for about 1.05 ms/token.  Moving the Qwen selector ahead
  of the backend branch passed a dedicated `_use_aiter=False` wiring test and
  preserved the control completion hash exactly, but full-service throughput
  regressed to `47.32--47.55 tok/s` from about `49.57 tok/s` (-4.4%).  The
  smaller single-wave Top-10 changes graph scheduling and lengthens the exposed
  router/MoE critical path even though its standalone work is lower.  The
  hypothesis that its 64-thread CTA footprint caused the loss was tested by
  launching the exact same wave0 math in a 256-thread/four-wave CTA, matching
  the old AOT block footprint; it still measured only `47.45--47.64 tok/s`.
  Both selector variants were reverted; this is intentionally not treated as
  a configuration bug to fix again without a new scheduling design.

The unoptimized-FP8 startup observed during this audit was configuration error,
not a regression: omitting `SGLANG_QWEN4_GFX90A_MQ4G128_ROUTED=1` retained the
43.80-GiB/GCD FP8 expert path and decoded at only about 17.4 tok/s after JIT.
The routed-MQ4 service uses 34.38 GiB/GCD and restores the 49.6 tok/s baseline.

## 2026-08-28: local-ID remap and LM-head tail

After the K160/N24 wave64 checkpoint reached 53.27 tok/s, the graph still paid
for generic advanced indexing to map each layer's ten global expert IDs onto
the rank-local EP4 table.  The source IDs are int32, but the generic index path
also introduced an internal int64/int32 copy.  A graph-safe 64-thread HIP
lookup now reads the same mapping table and writes int32 directly; it is gated
only by the routed MQ4G128 method flag and therefore does not change other MoE
backends or assume contiguous expert placement.  The oracle covered valid,
remote, negative, and out-of-range IDs exactly.  ABBA microbenchmarking gave
25.99 us for advanced indexing and 13.33 us for the HIP lookup (1.95x).

The TP4 local LM-head projection `[1,2560] x [2560,62080]` also bypassed the
existing wave64 selector.  Calling the same guarded BF16 GEMV from the direct
LM-head path reduced its ABBA median from 346.18 to 238.08 us (1.45x); the
generic matmul remains the fallback for unsupported shapes/devices.

Together these changes raised ten hot 256-token native-AR requests from a
53.27 tok/s median to 55.02 tok/s (+3.3%).  All ten retained completion hash
`bcc6f565ee3098ae`; the remap and routed-MQ4 oracles passed.  The cumulative
gain over the 49.57 tok/s pre-K160 baseline is about 11.0%.

Rejected follow-ups:

- MQ4 indexed CTA64 -> CTA256 preserved the per-row oracle but did not improve
  the dominant gate projection (40.78 vs 40.95 us); down improved only about
  2.1%, so the change was fully reverted.
- `--enable-single-batch-overlap` produced 54.70 tok/s versus 55.02 without
  SBO while preserving the hash; it is not enabled.
- A current runtime trace attributes roughly 4.2 ms/token to about 97 AIter
  custom all-reduces.  The `[11,2560]` shape seen in CPU events belongs to the
  11-token eager prefill, not BS1 graph replay; do not infer an 11x decode
  payload from it.  AIter's `AITER_GFX90A_AR_SMALL_BLOCKS=1/2` variants were
  faster in isolation but failed the four-rank numerical check.  SGLang's own
  custom AR was slower than AIter for the 5-KiB message in the repository
  comparison benchmark.  All temporary benchmark-source edits were restored.

## 2026-08-28: Qwen HC peer-reduce epilogue fusion rejected

A Qwen-specific prototype reused AIter's one-stage peer-read all-reduce and
folded the attention HyperConnection apply into its epilogue.  The existing
eight FP32 gate-dot partials were consumed directly, reducing the graph from
`gate + all-reduce + HC apply` to `gate + fused all-reduce/apply`.  A TP4 eager
oracle was exact against the separately reduced mathematical reference on all
four ranks, and host-synchronized micro latency appeared promising (about
246.6 us to 159.3 us without an end barrier).

The actual graph/service gate rejected it.  The initial kernel inherited the
DSV fused epilogue's missing end barrier and produced a stable but different
256-token hash (`1a8c2dccd3a72692` rather than the retained
`bcc6f565ee3098ae`) at about 54.44 tok/s trimmed.  Adding the ordinary AIter
one-stage `end_sync` lifetime rule and matching the normal H=2560 launch
geometry (three CTAs instead of four) kept graph capture stable, but measured
only 54.12 tok/s trimmed and retained the different hash.  This is below the
55.02 tok/s checkpoint, so the SGLang and AIter prototype was fully reverted.

Operational note: rebuilding this conda ROCm extension requires
`PATH=/opt/rocm/bin:...` and `CPATH=/opt/rocm/core-7.14/include`; the base-conda
`hipcc` wrapper otherwise misses `hipsparse/hipsparse.h` and
`thrust/complex.h`.

## 2026-08-28: native CDNA2 BF16 MFMA GEMV rejected for BS1

An isolated HIP prototype mapped M=1 GEMV onto
`v_mfma_f32_4x4x4bf16_1k`, populating only logical matrix row zero.  This is
the smallest CDNA2 BF16 MFMA padding factor (4x), and the empirically derived
lane mapping was correct: `(N,K)=(64,16),(320,2560),(4096,2560)` matched the
BF16 reference exactly in the initial oracle.

Preallocated-output, eight-pair ABBA showed why it must not replace the
existing wave64 GEMV.  Representative medians in microseconds were:

- `(320,2560)`: wave64 5.28, MFMA 146.69;
- `(4096,2560)`: wave64 21.46, MFMA 181.57;
- `(2560,160)`: wave64 5.53, MFMA 11.08;
- `(2560,1536)`: wave64 6.55, MFMA 101.44;
- `(1536,2560)`: wave64 6.77, MFMA 170.32.

The M=1 tile requires a serial dependency chain of one MFMA per four K
elements (640 instructions at K=2560), while the VALU wave64 kernel divides K
across 64 lanes and reduces once.  MFMA remains appropriate for higher-M
tiers, but direct BS1 substitution is structurally wrong.  The prototype was
fully reverted before any model service run.

## 2026-08-28: two-stage wave64 Qwen HC mix

The capture trace attributed about 4.90 ms/token to 97 Qwen hyperconnection
mix calls.  The existing Triton persistent kernel fused the full operation but
paid for FP32 atomics and two software grid barriers.  A gfx90a HIP replacement
splits at the natural 320-element low-rank boundary:

1. four-wave workgroups compute the `10240 -> 320` BF16-weight projection into
   FP32 without split-K atomics;
2. a second wave64 kernel applies scaled SiLU, computes all four `320 -> 2560`
   gates in registers, and fuses sigmoid, four-stream weighting, and the mean.

For the exact production shape, the pair measured 29.36 us trimmed versus
53.41 us for the persistent kernel, and was bitwise equal to both the old
kernel and an eager mathematical oracle.  The dedicated gfx90a test passed.
TP4/EP4 graph-BS1 native-AR ABBA (seven 128-token requests per arm) was:

- A1 legacy HC (prior committed service): 37.282 tok/s trimmed;
- B1 two-stage HIP HC: 42.436 tok/s trimmed;
- A2 same code with `SGLANG_QWEN4_GFX90A_HC_MIX_HIP=0`: 37.334 tok/s;
- B2 independent HIP HC service: 42.346 tok/s.

The mean-to-mean gain is about 13.7%.  Both B services passed the fixed France
oracle 10/10.  The exact-shape selector is default-on with a kill switch and
does not affect prefill, larger batches, other hidden sizes, or non-HIP GPUs.

## 2026-08-27: enable portable QSA radix top-512 on gfx90a

The decode capture trace showed that the QSA indexer still used the HIP
`torch.topk` fallback.  Across the 12 QSA layers its mask, radix sort, gather,
and associated elementwise chain cost about 1.30 ms/token.  The fallback was
selected by a stale `_is_hip` guard whose comment claimed the packaged JIT
kernel used PTX-only primitives.  In fact `fast_topk.cuh` uses SGLang portable
device helpers and compiled directly with hipcc on gfx90a.

For `[1,1024]` FP32 scores, length 769 and top-k 512, the JIT radix selector
was 14.58 us trimmed versus 65.01 us for mask plus `torch.topk`, 4.46x faster.
Candidate sets were exact.  The oracle covered B=1/4/8, L=512/513/1024/2048,
zero and nonzero row starts, and valid lengths both below and above 512.  The
QSA top-k/compressed tests passed 4/4; the larger suite passed its first 52
tests before an unrelated pre-existing fixture failed because its fake
metadata lacks `compress_member_rows`.

`SGLANG_QWEN4_GFX90A_QSA_JIT_TOPK=1` is now default and can be disabled for
same-code fallback comparisons.  TP4/EP4, AIter full attention, graph BS1,
native AR, seven 128-token rounds gave:

- prior old-path checkpoint: about 34.39 tok/s trimmed;
- B1 JIT top-k: 37.380 tok/s trimmed;
- A2 same code with the switch disabled: 34.637 tok/s trimmed;
- B2 independent JIT-top-k service: 37.282 tok/s trimmed.

The same-code A2/B2 gain is 7.64%; both B services passed the fixed France
oracle 10/10.  The final B2 service remained running on GCD 0--3.

The same investigation tested the full-attention backend.  This model has 12
full-attention and 36 Gated-DeltaNet layers.  A Triton-attention service was
correct (France 10/10) but measured 34.49 tok/s versus the then-current AIter
34.39 tok/s, only about 0.3%.  AIter already calls AMD's fused
`paged_attention_ragged`; NVIDIA FA3/FA4 is unavailable on gfx90a.  Full
attention backend replacement is therefore not a decode priority.

A selective BF16-shadow probe was also rejected.  Native wave64 BF16 GEMV was
very fast in isolation for Qwen projection shapes (for example 20.43 us for
4096x2560, 6.93 us for 1536x2560, and 4.65 us for 640x2560), but converting
only the 36 GDN QKVZ weights from block-FP8 to BF16 and selecting the wave64
kernel changed the service from 37.282 to 37.172 tok/s trimmed (-0.3%).  France
remained correct 10/10.  The likely cause is loss of the existing FP8 QKVZ/BA
dual-stream overlap; this again demonstrates that an isolated GEMV win is not
a graph critical-path win.  The shadow selector and K=2560 probe geometry were
removed.

## 2026-08-27: DSV4 in-block Q8 + `sdot4` transfer was negative

The gfx90a DSV4 FP4 M=1 kernel's most relevant pattern was reproduced as an
isolated MQ4G128 indexed oracle: each consumer block quantized the FP32
activation to symmetric Q8 in LDS, shifted the affine MQ4 nibble to `q-8`, and
used `__builtin_amdgcn_sdot4`.  The affine correction was
`sw*sx*dot(qw-8,qx) + (zero+8*sw)*sx*sum(qx)`.  The implementation matched the
FP32-activation indexed path within about 0.55% relative L2 (gate/up) and 0.51%
(down).

With the original two-output-row block, repeated activation quantization made
the kernels 81.20 vs 41.11 us (gate/up) and 51.60 vs 21.07 us (down).  Expanding
to 256 threads/eight output rows amortized the quantization, but remained
negative in nine-round trimmed timing:

- gate/up (`T=10,N=1280,K=2560`): 44.14 vs 41.07 us, 0.930x;
- down (`T=10,N=2560,K=640`): 29.10 vs 21.37 us, 0.734x.

This DSV4 optimization does not transfer directly.  DSV4 avoids expensive
nonlinear E2M1 codebook decoding, while MQ4G128 already has a cheap affine
nibble decode; Q8 conversion, LDS synchronization, and zero-point correction
cost more than `sdot4` saves.  The experimental entry point was removed and was
never wired into the production selector.

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

## Combine-produced RMS stats oracle (rejected)

An independent, production-disconnected oracle tested the exact BS1 boundary
`attn_hyper_connection.combine -> mlp_hyper_connection.mix`.  The existing
precomputed-gate combine apply uses eight CTAs, each covering half a 2560-wide
branch.  The oracle instead used four 160-thread CTAs, one per full branch.
Each thread applied two eight-element vectors, rounded the combined residual to
BF16, then accumulated both vectors before the same warp/CTA reduction tree as
`GroupedGemmaRMSNorm<2560>`.  It emitted four FP32 sums of squares alongside
the raw BF16 residual.

The following split-4 down kernel maps each K split exactly to one HC branch.
It read the raw residual, corresponding norm-weight slice, and branch sum of
squares; reconstructed the Gemma-normalized BF16 values in LDS; and retained
the existing wave64 down dot and fixed FP32 partial layout.  No normalized
`[1,10240]` tensor was materialized in the timed candidate.

Correctness passed all gates: combined raw BF16, a debug materialization of
normalized BF16, and split-4 down FP32 partials were all bitwise exact; 20
independent input seeds remained exact; and 1000 captured-graph replays kept
raw/partial outputs exact with no stale state.

Seven-round graph ABBA for the complete boundary was nevertheless neutral:

```text
A = current combine apply + grouped RMS + current split-4 down: 14.192 us
B = combine+sumsq + raw/norm direct split-4 down:             13.986 us
saved:                                                         0.206 us (1.45%)
```

The numerical design is valid, but every one of the 80 row-block CTAs per
branch reconstructs the same 2560 normalized values in private LDS.  The
repeated norm-weight load and elementwise work almost completely consume the
standalone grouped-RMS launch saving.  This is far below the production gate,
so the oracle was removed and no selector/service integration was attempted.

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

## HIP PLE fused-hash reachability and first-replay crash

`SGLANG_ENABLE_QWEN4_PLE_FUSION=1` did not actually select the fused N-gram
hash on HIP because `can_fuse_qwen4_ngram_hash()` contained an unconditional
`not _is_hip` guard. The fallback graph retained PyTorch
`cummax/where/gather` metadata kernels. On several independent services the
first post-capture request caused all four ranks to report an HSA hardware
exception in `at::native::vectorized_gather_kernel<16, long>`; a slower
experimental down kernel could accidentally mask the race by changing graph
timing.

The fused hash is integer-only Triton and was enabled on HIP. Random contexts
at B=1/4/16/32, including EOS in every boundary position, matched the eager
hash element-for-element. The TP4/EP4 graph then captured and completed ten
consecutive France requests with one exact output. Seven hot 128-token native
AR requests measured `34.350--34.413 tok/s`, median `34.383` and trimmed
`34.389`, essentially neutral versus the 34.329 checkpoint while removing the
first-replay crash and the fallback small-kernel chain. This is a correctness
fix rather than a claimed 5% performance checkpoint.

The same stale HIP guard also made the exact PLE gate/value and short-conv
state fusions unreachable. After enabling them, B=1/4/16 gate/value outputs
were BF16 bitwise equal to the eager chain; BF16 and FP16 short-conv inputs and
in-place state updates were bitwise equal for B=1/4/16. The full service again
passed France 10/10. A (hash only) trimmed `34.3885 tok/s` and B (all three
fusions) trimmed `34.3875 tok/s`, so the extra launch removal is neutral on the
current graph critical path. It is retained as exact graph simplification, not
as a performance claim.

An actual packed-weight MFMA prototype mapped 16 GEMV output rows to MFMA M
and used only output column zero. It ran correctly on gfx90a with relative L2
about `8.0e-4` for gate/up and `6.8e-4` for down, but the 15/16 unused MFMA N
columns plus nibble-to-FP16 construction made gate/up `55.68 -> 110.40 us` and
down `36.48 -> 45.76 us`. The prototype was removed. MFMA is not a viable BS1
drop-in until multiple useful columns can share one expert weight tile.

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

## Bitwise MQ4 consumer reduction and Qwen router Top-10

The post-HC/QSA graph trace showed that the routed-MQ4 tail was larger than the
two projection kernels alone: indexed gate/up was about 40.0 us/layer, indexed
down 21.8 us/layer, and the separate FP32 router multiply/reduction/cast chain
about 25--30 us/layer. Invalid EP4 assignments already return before weight
decode, so compacting remote slots was not the missing BS1 optimization.

An initial kernel fused the down projection itself with router weighting and
reduction. It changed the old 53.35 us standalone chain to 12.32 us and reached
45.41 tok/s, but rare expert-dot accumulation changes produced one-BF16-LSB
differences and changed the deterministic 256-token completion hash. It was
removed rather than accepted on the France-only oracle.

The retained kernel leaves the existing indexed down projection untouched and
only fuses its consumers. It reproduces ATen `Reduce.cuh`'s `vt0=4` order for
top-k 10: accumulators consume slots `0/4/8`, `1/5/9`, `2/6`, and `3/7`, then
combine in that order. A 1000-seed random oracle was BF16 bitwise exact. A
second wave64 kernel specializes the Qwen `[1,512]` BF16 softmax Top-10; it was
bitwise identical to the AIter/AOT kernel for 1000 random seeds and reduced the
standalone launch from about 14.85 to 11.84 us.

Service ABBA used TP4/EP4/no-A2A, native AR, graph BS1, 256 generated tokens:

- A2, both consumers disabled: trimmed `42.549 tok/s`;
- B2, exact weighted reduction only: trimmed `44.512 tok/s`;
- B3, exact weighted reduction plus wave64 Top-10: trimmed `44.609 tok/s`.

B3 is about 4.84% above A2. France was exact in 10/10 B3 requests, and every
A2/B2/B3 256-token completion had the same SHA-256 prefix `ca1ee446c43da638`.
The full MQ4 kernel test file passed 8/8. Both switches default on with explicit
kill switches. Several startup attempts ended in scheduler SIGABRT, including
an A arm with both new kernels disabled; clean retries captured and served.
No contemporaneous AMD-SMI, MCE, EDAC, or GPU-reset evidence identified this
as a hardware fault, so it remains an intermittent ROCm/JIT graph-startup issue.

## Graph-safe QSA packed attention for Qwen4-Exp MTP verify

The checkpoint contains one native Qwen4-Exp MTP layer, but a four-token
target-verify graph originally failed before serving.  On HIP, a single request
has four query rows during verification; the QSA selector therefore bypassed
the gfx90a BS1 packed-attention kernel and entered the eager portability
fallback.  That fallback executed
`cu_seqlens_k.detach().cpu().tolist()` inside graph capture, which is an illegal
device-to-host synchronization.  This was a software path-selection bug, not a
GPU or host-memory failure.

The gfx90a packed-QSA kernel now supports a static batch dimension.  Each
`(batch, head, split)` CTA reads its compact row bounds directly from the
device-side `cu_seqlens_k`; packed K/V remain compact rather than being padded
per request.  The reduction keeps separate `(batch, head, split)` partials.
Unit coverage now spans B=1/4 and valid lengths 1/64/193/512/1024/2048, plus a
real HIP graph capture/replay oracle: 13/13 tests pass.  A full TP4/EP4 service
then captured both the four-token target-verify graph and one-token draft graph.

This enables measurement but does not make the current MTP mode an accepted
performance checkpoint.  France passed 10/10 exactly, while fixed-input greedy
256-token trials were only about 38--44 tok/s and produced multiple completion
hashes.  Runtime logs reported average accepted lengths around 2.5--3.0 out of
four.  Native AR remains the correctness/performance reference until the MTP
metadata/sampling divergence is isolated; these numbers must not be reported as
native-AR throughput or as progress toward a verified 120 tok/s result.

A follow-up isolation found that
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'` did not
actually disable QSA index sharing.  `_qsa_index_share_requested()` read the
nested checkpoint `text_config=true` before the explicit top-level override.
The precedence is now top-level override first, nested checkpoint default
second, with a unit test for `nested=true, explicit=false`.  With the override
really active, fixed-input MTP still produced 6/6 distinct 256-token hashes and
about 42 tok/s, so QSA IndexShare is not the root cause of MTP divergence.

Further conservative isolation also failed to make the current MTP path an
acceptable checkpoint: ReplaySSM alone gave 6/6 hashes; disabling the three
Qwen alternate-stream paths reduced but did not eliminate the variants; using
RCCL instead of AIter custom all-reduce produced one dominant 64-token hash in
7/8 trials but still one divergent trial and fell to about 26 tok/s.  A runtime
Torch profiler combined with RCCL entered ROCm async-signal wait, so that trace
was discarded and the service was stopped cleanly.  In contrast, the native
AR control after the batched-QSA change was 5/5 hash-stable at 44.615 tok/s
trimmed, matching the pre-change 44.609 tok/s checkpoint.

## Qwen-wide wave64 BF16 decode linears

The native-AR graph trace was initially misclassified as FP8 from its CK kernel
name.  Correlating each dispatch to its CPU op showed BF16 inputs and weights;
for example `[1,2560] @ [4096,2560]^T`.  The checkpoint's
`modules_to_not_convert` also excludes ordinary attention, GDN, router, shared
expert, PLE, and indexer linears from FP8.  A proposed BF16 shadow cache was
therefore a no-op (unchanged 34.38 GiB model memory and 44.71 tok/s) and was
removed before commit.

The existing native wave64 BF16 GEMV was instead extended from `K % 1024 == 0`
to the actual wave-stride requirement.  Qwen widths `K=2560/1536` use
`unroll=1` and satisfy `K % 512 == 0`; older divisible-by-1024 shapes keep their
tuned unroll.  A Qwen-only marker routes unquantized, shape-compatible M<=4
linears before generic AIter/CK dispatch.  The selector supports diagnostic
modes 0=off, 1=all, 2=attention output, 3=attention input, 4=shared expert,
5=router, and 6=PLE/indexer; mode 1 is the validated default.

Six real Qwen shapes showed about 37--45% lower standalone wrapper latency.
The GPU oracle covered `(N,K)=(4096,2560),(512,2560),(2560,1536),`
`(3584,2560),(640,2560),(320,2560)` plus HIP graph replay: 7/7 passed at
BF16-appropriate `rtol=atol=2e-2`.

Native-AR service ABBA, fixed input IDs, 256 forced tokens:

- A1 (wave64 off): 44.615 tok/s trimmed;
- B1 (all compatible linears): 49.885 tok/s trimmed;
- A2 independent restart: 44.573 tok/s trimmed;
- B2 independent restart: 49.709 tok/s trimmed.

The mean-to-mean improvement is about 11.6%.  B1/B2 each had one unique hash
within every repeated 256-token run, although the hash differs from CK because
the wave reduction and MFMA reduction order are not bitwise identical.  This
was accepted under the explicitly allowed FP16/BF16 numerical-equivalence
criterion only after broader validation: France, 17x23, Chinese explanation,
Python factorial, and binary-search prompts each returned identical answers in
two repeats; two forced 1024-token generations had the same SHA-256
`db5ef0d3cd530c26...` and ran at 51.80/51.82 tok/s.  Mode 0 remains the exact
CK-order rollback.
### Rejected: FP16-weight `v_dot2_f32_f16` BS1 GEMV (2026-08-28)

An isolated wave64 HIP kernel converted the BF16 activation pairs to FP16 and
used CDNA2 `v_dot2_f32_f16` against an FP16 copy of each weight matrix.  An
eight-round preallocated ABBA comparison against the retained BF16 wave64
kernel gave the following median launch times (microseconds):

| `(N,K)` | BF16 wave64 | FP16 dot2 | dot2 / BF16 |
|---|---:|---:|---:|
| `(320,2560)` | 5.425 | 6.203 | 1.143 |
| `(4096,2560)` | 21.579 | 21.484 | 0.996 |
| `(2560,160)` | 5.344 | 5.769 | 1.080 |
| `(2560,1536)` | 6.649 | 6.776 | 1.019 |
| `(1536,2560)` | 6.894 | 6.501 | 0.943 |
| `(512,2560)` | 4.699 | 4.925 | 1.048 |
| `(640,2560)` | 5.300 | 5.388 | 1.017 |
| `(24,2560)` | 5.234 | 6.746 | 1.289 |

The only material micro win was the `1536x2560` shape (~5.7%); the large LM
head was effectively tied and most projections regressed.  Some shapes also
changed BF16-rounded output (`max_abs` up to 0.125) because BF16 weights are
not generally represented exactly by FP16.  This does not justify a second
FP16 weight cache or a model-level correctness risk, so the prototype was
removed without service integration.

## 2026-08-28: Qwen HC one-row wave geometry

The retained two-stage HIP HC kernel originally assigned two independent
output rows to each wave in both projections.  A complete `down_rows x
up_rows` scan over `{1,2,4,8}^2`, using preallocated outputs and eight timing
rounds, found that one row per wave minimized VGPR/independent-accumulator
pressure despite launching more CTAs.  All 16 variants were bitwise equal.
The combined standalone median improved from 28.587 us at `(2,2)` to 26.341
us at `(1,1)` (7.9%).  The dedicated gfx90a HC test also passed.

Two independent TP4/EP4 graph-BS1 services using the same AIter all-reduce
binary gave:

- `(1,1)`: 55.4376 tok/s trimmed over 12 256-token requests;
- `(2,2)`: 54.9573 tok/s trimmed over 12 256-token requests.

Both arms produced the same completion hash `1a8c2dccd3a72692` in every
round, so the retained default is a bitwise-safe ~0.87% service gain.  The
`SGLANG_QWEN4_GFX90A_HC_ROWS=2` override remains available for exact same-code
rollback/A-B checks; the default is one.

### Rejected: AIter direct/naive small-message all-reduce

A four-rank graph micro compared AIter `use_new=True` (shared-staging
one-stage) with its legacy `use_new=False` direct peer-read kernel for the
production 2560-element BF16 payload.  Eight-pair ABBA slowest-rank medians
were 12.024 us and 10.795 us respectively; outputs were bitwise equal.  A
temporary gfx90a/<=16-KiB selector then produced 55.6173 tok/s trimmed versus
the same-day 55.4376 baseline, only +0.32%, with all 12 completion hashes
equal to `1a8c2dccd3a72692`.

Profiling exposed why the micro win did not transfer cleanly: direct mode
shifted barrier waiting heavily between ranks (visible averages approximately
169 us on rank 0, 40 us on rank 1, and 17 us on rank 2).  Rank 3 failed to
write its DECODE trace; the profiler stop barrier then lost that peer and the
scheduler performed its normal SIGABRT cleanup.  There was no ECC/RAS, GPU
reset, VM fault, MCE, or EDAC evidence.  Given the negligible service gain and
less robust synchronization/profiling behavior, the selector was removed.

### Rejected: AIter shared-staging CTA thread-count scan

The normal one-stage kernel was templated experimentally at 128/256/512
threads while retaining rank order, FP32 accumulation, system-scope signals,
and the final buffer-lifetime barrier.  Four-rank graph ABBA for 2560 BF16
elements measured slowest-rank medians of approximately 12.94 us (128), 12.15
us (256), and 12.02 us (original 512).  All outputs were bitwise equal, but
smaller CTAs require more signal-bearing blocks and do not reduce the XGMI
critical path.  The AIter source and production `.so` SHA256
`5e8695f2d3da23eb...` were restored.

## 2026-08-28: fuse HC inject-gate production into HC down

Qwen's HC combine gate depends only on the normalized residual available
before the attention/MoE block, but the old graph launched its 8x4 partial-dot
kernel after the block's TP all-reduce.  The first eight CTAs of the retained
gfx90a HC down projection now reuse the same staged 10240-wide input and replay
the exact 8-split, 32-thread inject-weight dot decomposition.  The 32 FP32
partials travel in the residual tuple; after all-reduce, combine launches only
the apply kernel.  This removes one launch per attention/MLP sublayer without
changing BF16 rounding or the collective.

The standalone oracle was bitwise exact for the mixed input, all gate
partials, and final 10240-wide combined residual.  Ten-pair ABBA reduced the
combined mix+combine micro path from 32.578 to 30.983 us (~4.9%).  Full-service
ABBA, twelve 256-token requests per arm, was:

- A1 off: 55.4376 tok/s trimmed;
- B1 on: 55.8979 tok/s (+0.83%);
- A2 off: 55.6778 tok/s;
- B2 on: 56.1193 tok/s (+0.79%).

All 48 arm requests had completion hash `1a8c2dccd3a72692`.  Two longer runs
both stopped naturally at 502 generated tokens with identical SHA256
`c9413cf51a4ce9af...` and ran at 57.92/58.03 tok/s.  The dedicated gfx90a
tests cover the full bitwise oracle (2/2 passed).  The fusion is default-on
with `SGLANG_QWEN4_GFX90A_HC_GATE_FUSION=0` as an exact rollback.

The first service attempt also established an important scope guard: the
model-final HC mixer is constructed with `use_combine=False` and therefore has
no inject weight.  Gate fusion must require `block_inject_weight`; otherwise
startup fails with `AttributeError` and scheduler SIGABRT cleanup.  The guard
is retained.

### Rejected: HC apply + following grouped RMSNorm

A one-CTA-per-branch kernel reproduced HC apply's BF16 residual rounding and
then performed the immediately following MLP grouped RMSNorm in the same CTA,
emitting both residual and normalized residual.  Both tensors were bitwise
exact.  Ten-pair micro ABBA improved 12.600 us (two launches) to 6.844 us
(one launch), about 46% for this local pair.  Nevertheless, a full TP4 graph
service measured only 55.9869 tok/s trimmed, below the retained gate-fusion
checkpoint's 56.1193 tok/s, with the same completion hash.  Collapsing eight
160-thread apply CTAs plus four RMS CTAs into four longer CTAs worsened graph
tail scheduling enough to erase the standalone win.  All selector, model
plumbing, and kernel code was removed.

## 2026-08-28: Qwen LM-head geometry and rejected GDN launch scan

The TP4 LM-head shard has shape `[62080,2560]`.  A complete wave64 scan over
rows-per-wave `{1,2,4,8}` and waves-per-CTA `{4,8,16}` was bitwise exact for
all twelve variants.  The retained `(rows=1, waves=16)` median was 234.725 us
versus 239.784 us for the former generic `(2,8)` geometry, a 2.1% kernel win
or about 5 us per generated token.

The BS1 GDN recurrent core was also scanned at Triton `num_warps=1/2/4` and
`num_stages=1..4` for its real `[HV=48,V=128,K=128]` state.  More than one
warp preserved the immediate output but changed the persistent BF16 state by
up to 0.00390625, so it is not an exact default.  All one-warp stage counts
were bitwise exact.  A longer fourteen-pair ABBA rejected the apparent stage-4
micro win: stage 3/4 medians were 79.664/79.825 us (stage 4 0.2% slower).
Production remains one warp and three stages.

## 2026-08-28: Qwen compressed-QSA dense/sparse dual decode graphs

The twelve full-attention layers were still executing the complete compressed
indexer logits -> block Top-K -> token expansion chain even while every
visible token fit inside `indexer_budget=2048`.  The existing DeepSeek DSA
dual-graph mechanism was generalized, with model-scoped switches, to Qwen's
compressed QSA profile.  Each BS tier captures:

- `dense`: preserve indexer K projection/ring/compression state updates, but
  replace score/Top-K/expand with one kernel producing contiguous logical
  indices `[0, ..., visible-1, -1, ...]`;
- `sparse`: retain the complete original indexer for KV lengths above 2048.

Replay dispatch uses the existing host-side `seq_lens_cpu` mirror, so it adds
no D2H synchronization.  Mixed batches select sparse if any row is long.  The
dense-index oracle covered lengths 1/17/512/2048 exactly and its registered
tests passed 3/3 (including the existing expansion tests).

TP4/EP4/no-A2A, graph BS1, native-AR twelve-round service results first improved
from the retained 56.1193 tok/s checkpoint to 59.8776 tok/s trimmed (+6.70%).
A same-code kill-switch repeat later measured 60.2522 tok/s enabled versus
56.0671 disabled (+7.46%).  All completion hashes remained
`1a8c2dccd3a72692`.  A 2030-token fixed-input
test generated 64 tokens twice, crossing the 2048 boundary from dense to
sparse graph; both finished by length with identical SHA256
`82d5ee5b1069f91e...`.  Capture time increased from about 4.7 to 13.7 seconds
and graph memory by only about 0.01 GiB.  The feature is default-on for Qwen
compressed QSA only and can be disabled with
`SGLANG_QWEN4_GFX90A_QSA_DUAL_GRAPH=0`.

### Dense QSA K-only projection

The first dense graph removed indexer scoring but still ran the fused
`index_qk_proj`: four 128-wide query heads plus one 128-wide key head.  Since
the dense graph selects every visible token, its query, query RMSNorm and RoPE
were dead.  The retained path slices the BF16 projection's final key rows and
uses a shape-tuned wave64 GEMV `(rows_per_wave=1, unroll=1, waves=4)`, while
preserving the key ring/compression update.  It reduces the dense projection
from 640 outputs to 128 outputs.  The M=1/2/4 oracle is bitwise exact against
the key tail of the original joint wave64 projection (4 registered tests,
including dense indices, passed).

Native-AR twelve-round ABBA, TP4/EP4/no-A2A and graph BS1, measured:

- A1, joint QK: `60.2522 tok/s` trimmed;
- B1, K-only: `63.6097 tok/s` (+5.57%);
- A2, joint QK: `60.2996 tok/s`;
- B2, K-only: `63.4296 tok/s` (+5.19%).

All 48 requests produced 256 tokens with completion hash
`1a8c2dccd3a72692`.  Two K-only 2030+64 boundary runs again produced exact
SHA256 `82d5ee5b1069f91e...`, covering replay dispatch from dense to sparse at
KV length 2048.  The feature defaults on and has the exact rollback
`SGLANG_QWEN4_GFX90A_QSA_DENSE_K_ONLY=0`.

### Compression-phase dense graph

The fixed-shape QSA decode graph originally ran its full four-key compression
chain on every token.  Only sequence lengths divisible by the compression
ratio four complete a group; the other three phases computed gather, mean,
K RMSNorm, RoPE and cache store merely to write an inert reserved slot zero.
The runner now captures a third `dense_nocompress` graph.  Replay uses the
existing host `seq_lens_cpu` mirror: below the 2048-token dense budget it
chooses the normal dense graph if any row completes a group and otherwise the
non-compression graph.  The latter still writes raw K and RoPE position into
the pending ring.  Mixed batches conservatively compress if any row is on a
boundary; long rows still select the sparse graph.

The host selector oracle covered lengths 1/2/3/4/2048/2049 plus mixed
`[1,4]` and `[1,3]`.  Native-AR twelve-round ABBA measured:

- A1, compression every step (previous checkpoint): `63.4296 tok/s`;
- B1, phase graph: `68.4602 tok/s` (+7.93%);
- A2, compression every step: `63.5764 tok/s`;
- B2, phase graph: `69.4768 tok/s` (+9.28%).

Every request retained completion hash `1a8c2dccd3a72692`.  Two 2030+64 runs
exercised non-boundary dense, boundary dense, and sparse replay and retained
the exact prior SHA256 `82d5ee5b1069f91e...`.  The extra graph added only
about 0.01 GiB and roughly 0.7 seconds to capture.  The exact rollback is
`SGLANG_QWEN4_GFX90A_QSA_COMPRESSION_PHASE_GRAPH=0`.

### Correctness fix: short-extend QSA compression gather

An eager prefill with only three tokens (for example, `The sky is`) trapped in
HIP's vectorized gather with `HSA_STATUS_ERROR_EXCEPTION`.  This was a software
out-of-bounds access, not an ECC/MCE hardware failure.  The static QSA write
plan pads inactive compression groups with `member_rows=0`; the extend path
therefore still forms the inert gather `[0,1,2,3]`.  With fewer than four
source rows that gather exceeded `token_k`, even though its result was destined
only for reserved cache slot zero.

The retained fix pads only the transient extend source to the compression
ratio when the prompt has one to three rows.  Real completed groups already
contain at least four rows, so their numerical path is unchanged.  Registered
tests explicitly execute and synchronize the one-, two-, and three-row cases;
the focused QSA suite passed 7/7.  On the TP4/EP4/no-A2A service the formerly
crashing three-token request completed twice with identical token IDs.  The
256-token native-AR check retained hash `1a8c2dccd3a72692` at roughly
68.0--68.5 tok/s, and two fixed 2030+64 dense/compression/sparse boundary runs
both completed with identical SHA256
`bcb568e75fdc51e1d9b1434752248c8c9b785d9fa51b9a98e7abd13cdee1067d`.

### Rejected: cross-layer shared dense indices

Dense logical indices are layer-invariant, so an experiment generated the
graph-stable `[BS,2051]` tensor in the first full-attention layer and reused it
in the remaining eleven.  Short-request ABBA looked mildly positive:
`69.0500` and `69.1139 tok/s` enabled versus `68.6257 tok/s` disabled
(+0.62--0.71%), and all 256-token hashes matched.  The stronger 2030+64
dense/compression/sparse boundary repeat failed: the first run retained SHA
`82d5ee5b...`, while the second stopped at token 44 with a different hash.
This exposed an unsafe cross-layer graph tensor/stream lifetime despite the
short oracle.  The complete shared-buffer implementation and switch were
removed; per-layer dense-index generation remains the correctness baseline.

### Rejected: direct QSA K-ring GEMV epilogue

A wave64 specialization scattered the 128-wide dense K projection directly
to each dynamic pending-ring slot.  M=1/2/4 were bitwise exact; the M1 micro
reduced the Python-observed GEMV+`index_copy_` chain from roughly 66--68 us to
19.8 us.  An initial integration incorrectly reused the existing
`state_stored` flag (which means both K and RoPE position are stored) for the
K-only epilogue.  The short hash did not catch the stale-position state.  A
dedicated `key_stored` oracle exposed and fixed that semantic bug, keeping the
position write on the original path.

After the fix, native-AR service measured `68.9440 tok/s` enabled versus
`68.9734 tok/s` disabled, with identical completion hash and exact repeated
2030+64 boundary SHA.  The direct scatter did not transfer through the
alternate-stream graph schedule.  Fusing the three-axis position write into
the GEMV also failed (`69.1820 tok/s`, below the earlier K-only experimental
run).  Both epilogues, their switch, tests and JIT code were completely
removed.

### Rejected: unnormalized gfx90a router Top-K

Qwen sets `norm_topk_prob=None`, so the custom gfx90a Top-K selector had been
unreachable even though the kernel supported the production E512/K10 shape.
An experimental unnormalized specialization matched AIter IDs and weights
exactly and reduced the standalone launch from 13.711 to 11.606 us (~15.4%).
The full dual-graph service, however, measured 60.1015 tok/s versus the
retained 60.2522 tok/s baseline, with the same completion hash.  The selector
and kernel extension were reverted: the shorter kernel did not improve the
graph's arrival/synchronization critical path.

### Rejected: indexed MQ4 gate/up plus SwiGLU HIP epilogue

A shape-locked gfx90a HIP oracle paired the two 32-lane subgroups of one
wave64: subgroup 0 computed `gate[row]`, subgroup 1 computed `up[row+640]`,
both retained the production `mq4g128_dot_row` FP32 accumulation order, and
the epilogue emitted `F.silu(gate) * up` directly as `[1,10,640]` FP32.  Both a
width-64 shuffle exchange and an 8-byte LDS exchange were tested.  Neither
needed LDS for correctness; both matched the registered indexed projection
plus torch `F.silu` bitwise for valid and `-1` remote expert IDs.  LDS was
roughly 1--2 us faster in the co-resident micro.

Real `E128/M1/T10/N1280/K2560` interleaved microbenchmarks measured the old
indexed projection plus SwiGLU at roughly `111--135 us`, versus `73--82 us`
for the fused HIP epilogue: a local 31--40% improvement, or approximately
`36--52 us/layer`.  The production candidate then ran the existing standalone
FWHT before down projection.  Its pre-FWHT output remained bitwise exact to
torch; compared with the retained Triton SwiGLU+FWHT boundary, the rotated
tensor differed only at the last FP32 bits (`max_abs=1.862645e-9`).

The guarded TP4/EP4 BS1 service B1 passed correctness: five hot 256-token runs
all retained the same completion hash.  Throughput was
`68.79 / 68.42 / 68.43 / 68.31 / 68.42 tok/s`, indistinguishable from the
roughly `68.0--69.5 tok/s` baseline.  The extra standalone FWHT launch and the
decode graph's scheduling/critical path absorbed the module-level saving.
The HIP kernel, default-off switch, production wiring, tests and benchmark
were therefore completely removed; this path must not be reintroduced based
on the standalone percentage alone.

### Rejected: TP4/EP2 MQ4G64 module path

The TP4/EP2 layout was audited and prototyped without loader or service
integration.  With MoE-DP1, `moe_tp_size=4/2=2`; each EP rank stores 256 of the
512 experts and each stored expert has logical intermediate shard `640/2=320`.
The MoE-TP groups are `[0,1]` and `[2,3]`, while MoE-EP groups are `[0,2]` and
`[1,3]`.  Standard dispatch maps the other 256 experts to `-1`; Qwen's retained
global TP4 all-reduce sums the EP and TP partials together after the separate
shared-expert partial is added.

The checkpoint's FP8 scales are global `[128,128]` blocks: gate/up weights are
`[640,2560]` with scales `[5,20]`, and down is `[2560,640]` with scales `[20,5]`.
The second TP shard begins at 320, halfway through global block 2.  Generic
AIter padding cannot load it correctly: padding local 320 to 384 makes the
existing loader slice rank 0 as `0:384` and rank 1 as `384:640 + zeros`, instead
of the required `0:320` and `320:640`.  A real loader would need logical-320
weight slicing plus offset-aware scale expansion; rank 1 consumes the tail of
block 2 and then blocks 3/4 for both w13 rows and w2 columns.

Two storage designs were compared.  G128 needs only down-K padding 320 to 384,
giving packed w13 `[256,640,20,72]` and w2 `[256,2560,3,72]`: 360 MiB/layer or
16.875 GiB for 48 layers/GCD.  G64 needs no padding and gives
`[256,640,40,40]` plus `[256,2560,5,40]`: 375 MiB/layer or 17.578 GiB/GCD.
For reference, retained TP4/EP4 G128 routed storage is 337.5 MiB/layer or
15.820 GiB/GCD.  Thus G64 costs about 1.758 GiB/GCD over EP4 and 0.703 GiB over
the padded-G128 EP2 design.  At mean occupancy, useful scalar MAC count is the
same as EP4, but G64 doubles group iterations/scale-zero metadata and reads
about 11.1% more packed bytes; padded G128 instead performs about 6.7% extra
down work.

The module oracle added FWHT64, affine G64 pack/dequant, a 32-lane G64 indexed
HIP dot, and a pure global-coordinate FP8-slice converter.  Synthetic focused
tests passed 6/6, including valid/`-1` expert IDs and true gate/down dimensions;
the retained G128 regression subset passed 5/5.  Raw layer-0 checkpoint tests
for gate, up and down at offsets 0 and 320 all produced byte-identical G64 packs
against full-checkpoint dequantize-then-slice (6/6).

All microbenchmarks ran on GPU 7 only after `amd-smi process -g 7` reported
`No running processes detected`.  Projection-only shapes were EP4 G128 gate
`E128/M1/T10/N1280/K2560`, down `E128/M10/T1/N2560/K640`; EP2 G64 gate
`E256/M1/T10/N640/K2560`, down `E256/M10/T1/N2560/K320`.  Invalid slots stayed
in the fixed T10 extent.  Gate+down medians by live local assignments were:

- EP4 G128 live 3/4/5: `116.639 / 122.639 / 127.439 us`;
- EP2 G64 live 5/6/7: `123.199 / 132.240 / 129.998 us`.

The realistic critical-rank comparison, roughly EP4-live4 versus EP2-live6,
was about 7.8% slower for G64; even the high-tail live5 versus live7 comparison
was about 2.0% slower.  Since the no-padding design increased memory and did
not improve the module critical path, all G64 code, tests and benchmark were
removed.  The path must not proceed to loader/service integration without a
different kernel organization that first demonstrates a clear module win.

### Rejected: factorized-affine G128 dot

An independent indexed HIP oracle retained the production packed G128 layout
and 32-lane final reduction, but changed each lane's inner arithmetic.  For
each 128-element group it accumulated `q*x` and `sum(x)` separately, then added
`scale*qdot + zero*sumx` once per group.  The production kernel remained
untouched and the factorized helper had no service call site.

Correctness used the real TP4/EP4 projection shapes with three valid local
assignments and the remaining fixed T10 slots set to `-1`: gate
`E128/M1/T10/N1280/K2560` and down `E128/M10/T1/N2560/K640`.  Against explicitly
dequantized packed FP32 weights, factorized gate had maximum absolute error
`1.04308128e-7` and down `4.28408384e-8`.  Against the production indexed
kernel, maximum differences were `1.1920929e-7` and `3.7252903e-8`
respectively; invalid assignments remained exactly zero.  The changed affine
factorization is therefore numerically close but deliberately not bitwise.

GPU 7 was used only after `amd-smi process -g 7` reported no running processes.
Twenty-one interleaved ABBA rounds measured production gate/down medians of
`66.000 / 66.320 us` and factorized medians of `67.200 / 64.960 us`.  The
combined projection time changed from `132.319` to `132.160 us`, only a
`0.121%` speedup, far below the 8% module gate.  Extra qdot/sum-x dependency
chains cancel the saved affine operations, and gate is slightly slower.  The
oracle kernel, wrapper and benchmark were fully removed without service
testing.

### Retained: BS1 BF16 shared gate/up + SwiGLU subgroup fusion

An isolated HIP oracle tested the real TP4 Qwen shared-expert shape
`x=[1,2560]`, gate/up weight `[320,2560]`, and output `[1,160]`.  The reference
chain used the retained `gfx90a_wave64_bf16_gemv` into a preallocated BF16
`[1,320]` buffer followed by SGLang's ROCm `SiluAndMul`.  Candidate arithmetic
explicitly retained both rounding boundaries: FP32 dot to BF16 gate/up, FP32
SiLU to BF16, then BF16 multiply to BF16 output.

The 512-thread prototype assigns the lower and upper 32 lanes of each wave to
gate and up.  Changing the reduction width can change the final BF16 result:
the worst observed case differed in one of 160 outputs by
`1.39698386e-9` (`mean_abs=8.731149e-12`).  This is not bitwise exact, but was
accepted as a bounded candidate after explicit review.

Two bitwise-exact wave64 variants were then tested.  The first used two
accumulators per wave and exactly the old lane-to-K mapping and offset
`32 -> 1` reduction tree.  The second used 16 independent wave64s per block,
with separate gate/up waves and a shared-memory pairing boundary.  Both passed
four random seeds bitwise and a CUDA graph capture/replay test.  The temporary
registered tests reported `2 passed` before the experiment was removed.

The apparent single-run gain of the dual-accumulator exact version did not
survive independent repetitions.  Three subsequent 31-round ABBA processes measured
fused versus the complete old chain at `-16.4%`, `-5.3%`, and `+2.1%`.  The
16-wave version measured `-3.0%`, `-6.9%`, and `-2.9%` in three independent
runs.  Exact wave64 pairing is therefore rejected.

The subgroup candidate was then rerun with four independently allocated tensor
sets and seeds, each using 31 interleaved ABBA rounds after warmup.  Median
complete-chain speedups were `52.81% / 51.15% / 49.17% / 51.13%`; trimmed-mean
speedups were `49.49% / 49.58% / 50.40% / 52.56%`.  The first three seeds were
bitwise exact and the fourth produced only the single bounded difference above.
One thousand captured graph replays remained finite and bitwise stable.
Registered multi-seed tolerance and 1000-replay tests passed `2/2`.

All GPU runs used GCD 7 only after `amd-smi process -g 7` reported no running
processes.  The subgroup HIP kernel, JIT wrapper, benchmark and an exact-shape
Qwen shared-MLP call site are retained behind the default-on
`SGLANG_QWEN4_GFX90A_FUSED_SHARED_GATE_UP_SWIGLU` selector.  The call site also
requires HIP, an unquantized bias-free gate/up layer, `x=[1,2560]`, BF16
contiguous tensors, local weight `[320,2560]`, and gfx90a; every other case
falls back to the old projection and activation chain.

TP4/EP4/no-A2A graph-BS1 service A/B used six independent 256-token requests
per arm after all lazy JIT modules were warm.  Enabled hot rates were
`69.91 / 70.20 / 70.16 / 70.11 / 69.87 tok/s` (median about `70.11`); the
same-code disabled rates were
`68.99 / 69.13 / 69.35 / 69.25 / 69.10 tok/s` (median about `69.13`).  Thus
client wall-time improved about 1.42%.  The server steady decode window moved
from about `75.3` to `76.4 tok/s`, independently confirming about 1.46%.
Every 256-token completion retained hash `1a8c2dccd3a72692`.  France remained
semantically correct, and two fixed 2030+64 dense/compression/sparse boundary
runs both retained exact SHA256
`bcb568e75fdc51e1d9b1434752248c8c9b785d9fa51b9a98e7abd13cdee1067d`.
The feature therefore defaults on, with `=0` as the exact fallback.

### Rejected: stock AIter all-reduce plus RMSNorm fusion

The existing `--enable-aiter-allreduce-fusion` path was tested after the
shared-expert gate/up+SwiGLU checkpoint because it can remove the standalone
post-collective norm launch without changing the model weights.  TP4/EP4,
no-A2A, graph-BS1 service A/B used identical fixed 64-token input IDs and
forced 256-token native-AR completions.  Excluding each arm's warmup, the
trimmed client rates were `69.760 tok/s` disabled and `69.999 tok/s` enabled,
only `+0.34%`.  The corresponding steady server windows were approximately
`76.3--76.4` and `76.6--76.8 tok/s`.  Every request in both arms produced the
same completion SHA-256 prefix `ac55ed9f7239753d`.  The stock fusion is
therefore numerically safe here but remains disabled: its launch saving does
not survive the graph critical-path schedule strongly enough to justify a new
default.

This experiment also exposed an important launch-profile distinction.  The
accepted MQ4 service must keep import-time `SGLANG_USE_AITER=0`; AIter custom
all-reduce is selected independently by its default-on AR integration.  Setting
global `SGLANG_USE_AITER=1` changes Qwen MoE construction, makes the routed
MQ4 shape selector unreachable, and restores the FP8 expert footprint from
`34.38` to `43.95 GiB/GCD`.  Such a run is not comparable to the accepted
MQ4 profile.  Every GPU launch in this experiment was preceded by a clean
`amd-smi process --json` scan.

## 2026-08-28: occupancy-oriented HC down split-K

The BS1 HC down projection has shape `10240 -> 320`.  The retained one-row
wave64 kernel launched only 80 four-wave CTAs, occupying about 26% of the 304
CUs on one MI250 GCD.  An atomic-free split-K specialization now divides each
row's contiguous K traversal into four parts, expands the grid to 320 CTAs,
writes static FP32 partials, and combines them in split order with one fixed
reduction CTA.  The existing eight-way inject-gate decomposition remains on
the split-0 CTAs and is bitwise unchanged.  `SGLANG_QWEN4_GFX90A_HC_DOWN_SPLIT`
accepts 1/2/4/8; split 4 is the gfx90a default and split 1 is the rollback.

Forty-one-round module ABBA measured the complete down + up + gate-production
path at `29.13 us` for split 1, `25.18 us` for split 4, and `25.36 us` for
split 8.  Thus split 4 improves the full HC mix by about 13.6%; split 8 was
rejected.  Four random seeds produced bitwise-identical BF16 mixed outputs and
gate partials, and 1000 captured graph replays were finite and bitwise stable.
The registered gfx90a HC suite passed 3/3.

TP4/EP4/no-A2A graph-BS1 native-AR service ABBA used fixed 64-token input IDs,
forced 256-token completions, twelve hot samples per arm, and two-sample trim:

- A1 split 1: `69.760 tok/s`;
- B1 split 4: `74.531 tok/s`;
- A2 split 1: `71.894 tok/s`;
- B2 split 4: `74.669 tok/s`.

The arm means improve from about `70.827` to `74.600 tok/s`, or 5.33%; the
neighboring A2/B2 pair is independently positive by about 3.86%.  All 50
service requests retained completion SHA-256 prefix `ac55ed9f7239753d`.
France returned `Paris` exactly twice.  Two 2030+64 dense/compression/sparse
boundary runs retained identical SHA256
`bf1d164575a4bb9f1a58bd25a42a523627ea94a781bea887db73691796175e27`.

## Rejected gfx90a GDN ratio-8 fused split

The production MQ4 configuration disables the global AIter switch, which made
Qwen3.5 GDN ratio 8 use the fallback `split + cat + contiguous` path even though
the existing fused split kernel supports this geometry.  A guarded experiment
made ratio 8 reachable on HIP independently of `SGLANG_USE_AITER`.  Its focused
ratio-8 test was exact, so this was not a correctness limitation.

End-to-end restart measurements did not reproduce a useful gain: fused runs
were `76.408` and `76.204 tok/s`, while the intervening fallback control was
`76.231 tok/s`.  The apparent first-run improvement disappeared on repetition.
The selector and test were therefore fully reverted.  A multi-stream profiler
attempt also stalled during profiler shutdown with the shared/routed alternate
stream active; that trace was discarded and all involved processes were killed.
Every service/GPU experiment was preceded by `amd-smi process --json`.

The split-K reduction was subsequently folded into the first phase of the HC
up kernel.  Its first 320 threads now sum the four partial rows in the same
`p0+p1+p2+p3` order before applying SiLU, eliminating the standalone reduction
CTA and its 320-element intermediate write/read.  The complete split4 HC mix
improved from about `25.74` to `23.19 us` in 41-round ABBA (9.9%), remained
bitwise equal, and the 3-test suite still passed.  Service A/B used the same
fixed input and twelve hot 256-token samples:

- B1 fused into up: `75.024 tok/s`;
- A2 standalone reduction: `74.565 tok/s`;
- B2 fused into up: `75.131 tok/s`.

Together with the preceding standalone-reduction B2 (`74.669 tok/s`), the two
arm means are approximately `75.077` versus `74.617 tok/s`, a small but stable
0.62% gain.  All 39 new requests retained hash `ac55ed9f7239753d`.
`SGLANG_QWEN4_GFX90A_HC_SPLIT_REDUCE_IN_UP=0` restores the standalone
reduction; the fused form defaults on.

An isolated HC-up weight-only INT8 prototype was also rejected.  Per-row
symmetric INT8 weights reduced the real `10240 x 320` up kernel only from
`11.21` to `10.97 us` (about 2.1%) while already introducing max BF16 output
error `0.00390625` and relative L2 error `7.8e-4`.  The full-model ceiling is
well below 0.2%, so neither the kernel nor an INT8 shadow cache was retained.

An isolated full-HC FP16 `v_dot2_f32_f16` oracle was likewise rejected.  It
kept the production split-4 grid, inject-gate decomposition and final mixing,
but stored both HC projection matrices in FP16 and used CDNA2 packed dot2.
Twelve-round graph ABBA improved the complete mix only from `24.736` to
`23.944 us` (3.20%).  The gate partials remained bitwise exact and 1000 graph
replays were stable, while the mixed BF16 output had `max_abs=4.8828125e-4`,
relative L2 `2.30e-4`, and cosine `0.99999994`.  The model-level ceiling is
roughly 0.4--0.6% but an FP16 shadow would consume about 1.2 GiB/GCD, so all
oracle code was removed.

The Qwen dual-stream MoE schedule was also tested by swapping branch roles:
routed gate/Top-K/MQ4 remained on the graph current stream while the shared
expert ran on the side stream.  The plain swap was neutral (`74.972 tok/s`
trimmed, exact fixed-input hash).  Giving the swapped shared side stream ROCm
high priority produced a promising B1 `75.641`, but failed restart ABBA: A2
was `74.924` and B3 was `74.776 tok/s`.  The candidate/control means differed
by only about 0.28% and the sign reversed across restarts.  Both scheduling
switches were removed; stream identity/priority alone does not resolve the
routed/shared CU contention.

## BS1 EP4 persistent-slot MQ4 down projection

The indexed MQ4 grid previously included the full static Top-10 dimension.
After EP4 remapping, a rank normally owns only two or three assignments, but
the old `(N/2, assignment)` grid still scheduled CTAs for all ten and returned
early for remote IDs.  A persistent-slot kernel now launches only `N/2` CTAs;
each row CTA walks the ten assignments, computes valid local experts, and
writes exact zeros for remote assignments.  This is restricted to the
production down layout `M*T=10, K=640`.  Applying it to gate/up K=2560 was
correct but 9.1% slower because serializing its longer dots lost assignment
parallelism.

For the real down shape `(M,T,N,K)=(10,1,2560,640)` with three valid local
assignments, ten-round graph medians improved `20.788 -> 11.644 us` (44.0%).
The FP32 partial difference was bounded by `3.58e-7`, relative L2 `9.21e-8`;
remote outputs remained exact zero, the dedicated oracle passed, and 1000
graph replays were deterministic.  Runtime service trace confirmed production
selection and improved the contended kernel from about `15.2` to `9.2 us`
per layer.

Native-AR TP4/EP4/no-A2A fixed-input restart ABBA, 256 forced tokens:

- controls A2/A3: `74.924 / 74.488 tok/s` trimmed;
- candidates B1/B2: `75.921 / 75.546 tok/s` trimmed;
- arm means: `74.706 -> 75.734 tok/s`, about `+1.38%`.

All measured requests retained completion hash `ac55ed9f7239753d`.  France
returned exactly `The capital of France is Paris.` twice with normal stop, and
the 2030+64 boundary retained text SHA-256
`bf1d164575a4bb9f1a58bd25a42a523627ea94a781bea887db73691796175e27`.
The path defaults on for the guarded BS1 K640 selector;
`SGLANG_QWEN4_GFX90A_MQ4G128_PERSISTENT_SLOTS=0` restores the old grid.

A follow-up fused router-weight reduction into the persistent-down CTA was
bitwise BF16 exact and removed the `[10,2560]` FP32 partial write/read plus one
kernel launch.  Register and LDS variants both reduced isolated graph latency
only from about `17.92` to `15.58 us` (13.1% for that two-kernel chain):
carrying the reduction state lengthened the persistent dot kernel itself.
The service measured `75.656 tok/s`, inside the retained persistent-down
restart range `75.546--75.921`; no end-to-end gain was distinguishable.  The
fusion and its selector were removed.

A BS1-only lazy global-to-local expert mapping probe was also rejected.  It
kept global Top-10 IDs in the dispatcher and subtracted the rank's compile-time
contiguous expert offset inside indexed gate/up and persistent down; prefill
and grouped paths retained the ordinary mapping kernel.  Per-rank offset
oracles were exact, and runtime trace confirmed that decode remap launches
disappeared (the remaining 48 belonged to prefill).  Service B1 measured
`76.088 tok/s` versus an independent rollback A2 `75.929 tok/s`, only 0.21%.
This is below restart noise and does not justify dual global/local ID semantics,
so the dispatcher and kernel changes were fully removed.

## Restore only Qwen shared/routed alternate-stream overlap

The post-persistent runtime trace exposed that the current formal launch had
silently dropped all historical alternate-stream variables: every GDN and MoE
kernel was serialized on one stream.  Re-testing the switches independently
showed that only the shared/routed MoE overlap remains useful on the current
kernel stack:

- `SGLANG_ALT_STREAM=1` only: `76.264 tok/s` trimmed;
- plus `SGLANG_GDN_QKVZ_BA_ALT_STREAM=1`: `75.668 tok/s`;
- plus `SGLANG_QK_NORM_ALT_STREAM=1` instead: `75.681 tok/s`;
- all three historical switches: `75.205 tok/s`.

All arms retained fixed hash `ac55ed9f7239753d`.  The routed-MQ4 HIP profile
now defaults the base alt stream on when `SGLANG_ALT_STREAM` is absent, while
the two GDN/QK sub-overlaps remain off.  Explicit `SGLANG_ALT_STREAM=0` is the
rollback.  A no-explicit-alt restart confirmed the default (`75.859 tok/s`
trimmed), returned the exact France sentence twice, and retained the 2030+64
boundary SHA-256
`bf1d164575a4bb9f1a58bd25a42a523627ea94a781bea887db73691796175e27`.
