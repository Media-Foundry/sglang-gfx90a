# TP4 M32 logical-contiguous E8M0 scale oracle (2026-08-30)

## Audit and design

An older direct-MoE experiment tested row-major FP4 weights plus row-major
scales and only reported cosine-level agreement. It is not an exact precedent
for the current grouped TP4 path. This oracle leaves packed FP4 weights
unchanged and copies only the CK A16W4 shuffled E8M0 bytes into logical:

- gate/up: `[E, 2I, K/32]`, 32 MiB per layer;
- down: `[E, N, K/32]`, 16 MiB per layer.

A double-resident experiment therefore costs 48 MiB/layer or 2.016 GiB/GCD
for 43 layers. A final replacement layout would have no net scale-byte growth,
but would require load-time conversion. Candidate kernels differ only in using
the straight logical offset; packed weights, LUT decode, SDOT, FP32 addition,
DPP/reduction order and the same-group R2 prefetch schedule are unchanged.

The inverse CK layouts are exact tensor permutations:

- gate physical axes `[E,N1,K1,klane,nlane,kpack,gate_up]`;
- down physical axes `[E,N1,K1,klane,nlane,kpack,npack]`.

## Correctness and ABBA

The real diverse pass37/layer34 route was used. Across 100 activation and
router-weight mutations, gate BF16 intermediate, INT8 values/scales, down FP32
partial and final BF16 output were all bitwise exact.

Seven-round ABBA trimmed means:

| stage | shuffled scale | logical scale | delta |
|---|---:|---:|---:|
| gate/up | 244.689 us | 255.154 us | +10.465 us |
| quant | 42.713 us | 42.843 us | +0.130 us |
| down | 169.005 us | 159.684 us | -9.321 us |
| reduce | 3.708 us | 3.687 us | -0.021 us |
| full routed | 422.500 us | 426.920 us | +4.419 us |

Combined logical scales regress the full routed path by 1.035%. The CK shuffle
is not merely address arithmetic: for gate/up it coalesces the wave's R2 and
gate/up scale access better than logical row-major layout. Conversely, logical
down scales improve the subgroup16 same-row consecutive-group access by 5.84%.

Reject the combined layout and do not connect it to production. A separate
down-only logical-scale oracle is technically promising, but must retain
shuffled gate scales and prove that its ~9-us micro saving survives the full
multistream service before accepting the 16-MiB/layer conversion/cache cost.

Standalone changes only:

- added optional logical-scale template mode to the gate/down row-prefetch
  oracle headers;
- added `scripts/rocm/bench_dsv4_tp4_logical_scale_layout_oracle.py`.

## Down-only logical-scale follow-up

The same oracle added a third profile which retains CK-shuffled gate/up scales
and converts only down scales to logical `[E,N,K/32]`. Testing again began with
an empty-GPU `amd-smi process` check. All three profiles passed 100 activation
and router-weight mutations with exact intermediate BF16, INT8 values/scales,
FP32 partial and final BF16 output.

Seven-round ABBA trimmed means for down-only logical scales:

| stage | all shuffled | logical down only | delta |
|---|---:|---:|---:|
| gate/up | 244.859 us | 244.968 us | +0.109 us |
| quant | 42.624 us | 42.966 us | +0.342 us |
| down | 169.767 us | 160.461 us | -9.305 us |
| reduce | 4.285 us | 4.123 us | -0.162 us |
| full routed | 423.053 us | 417.636 us | -5.417 us |

Thus down-only improves the down kernel by 5.80% and the complete routed micro
by 1.30%. Its duplicate cache costs 16 MiB/layer, or 688 MiB/GCD for 43 layers.

The minimal eventual production connection point is in
`Fp8MoEMethod.process_weights_after_loading_block_quant`: for the strict
gfx90a DSV4 CKTile shape, clone `w2_weight_scale_inv` after
`_gfx90a_cktile_reorder_w2_rows` but before `shuffle_scale`. That tensor already
has the logical row order corresponding to the reordered packed w2 weights.
Expose it as an optional field of `AiterMoeQuantInfo`, and only substitute its
pointer when the existing strict TP4/M32 DPP/down-row-prefetch selector is true.
All other AIter/CK paths must continue consuming the shuffled scale. This is an
assessment only; no production cache or selector was added in this experiment.

## Default-off production experiment wiring

A later static-only change prepared the down-only experiment for service A/B:

- `SGLANG_DSV4_GFX90A_M32_LOGICAL_DOWN_SCALE` is default false globally and
  explicitly remains `0` in the TP4 BS32 profile unless the caller opts in.
- During FP4 post-load processing, only the existing strict gfx90a DSV4 CKTile
  shape clones the reordered logical W2 scale before the normal CK
  `shuffle_scale`. The clone is required to be uint8 `[256,4096,16]` and
  contiguous; otherwise explicit opt-in fails loudly.
- `AiterMoeQuantInfo` carries the optional logical pointer. The runner validates
  it again and substitutes it only after the existing exact TP4/M32 DPP plus
  down-row-prefetch selector and the final D832 condition are both true.
  Every other tier, MFMA path, down-consumer path and grouped fallback continues
  to receive the original shuffled scale.
- A separate cached JIT specialization instantiates the existing row-prefetch
  kernel with `kLogicalScale=true`; the default specialization is unchanged.

`py_compile`, `bash -n` and `git diff --check` passed. No service or additional
GPU test was run for this wiring, and it was not committed; production
teacher-forced and throughput validation is delegated to the reviewing agent.

## R2-packed logical down-scale oracle

The accepted logical-down micro was extended with a scale-only layout
`[E,N/2,K/32,2]` (passed through the existing equal-numel tensor contract).
For each same-group R2 task, the low/high E8M0 bytes are fetched by one aligned
`uint16` load and extracted in registers. Packed FP4 weights, activation loads,
LUT decode, SDOT, FP32 accumulation and the fixed reduction order are unchanged.
The byte count remains 16 MiB/layer.

Testing began with `amd-smi process` showing all eight GCDs idle. On the real
pass37/layer34 route, 100 activation/router-weight mutations were bitwise exact
through gate BF16, INT8 values/scales, down FP32 partial and final BF16 output.

Seven-round ABBA trimmed means:

| stage | CK shuffled | logical down | R2-packed logical down |
|---|---:|---:|---:|
| gate/up | 244.708 us | 244.797 us | 244.705 us |
| quant | 42.122 us | 42.316 us | 42.375 us |
| down | 169.459 us | 160.213 us | 159.662 us |
| reduce | 3.839 us | 3.755 us | 3.870 us |
| full routed | 422.248 us | 416.720 us | 415.754 us |

R2-packed scales improve down by 6.14% and full routed by 1.56% versus CK
shuffled scales, passing the >=0.5% continuation gate. Relative to ordinary
logical-down, the extra packing saves about 0.55 us in down and 0.97 us in the
full measurement. This remains an independent oracle only; no production
cache, selector or load-time conversion was changed for R2 packing, and no
commit was made.

## Production validation

The first two fail-loud startup attempts exposed two load-time assumptions
without executing an incorrect graph: direct FP4 mode does not enter the
CKTile W2-row-reorder condition, and the checkpoint scale tensor is typed
`float8_e8m0fnu` before AIter's shuffle. The accepted loader therefore captures
the direct profile's checkpoint-order scale immediately before shuffle and
bit-views the E8M0 payload as uint8 for the HIP kernel.

The corrected service captured graph tiers 1--32 successfully. Its 32-distinct
teacher-forced record was JSON-identical to the accepted issue-order-3 oracle,
including output IDs, full output logprob rows and top-5 entries. Five diverse
512-token runs with logical down scales measured
`628.668/628.315/629.111/629.411/627.703 tok/s` (median `628.668`, trimmed mean
`628.698`). A fresh same-HEAD independent control service measured
`620.921/624.157/623.861/624.565/625.078 tok/s` (median `624.157`, trimmed mean
`624.194`). All requests completed 512 tokens with `finish=length`, and every
round passed the France first-nine-token check.

The independent-service gain is 0.72% by both median and trimmed mean, in the
same direction as the exact routed micro's 1.30%. Enable the cache by default
only in the strict TP4 BS32 profile; the global environment default remains
false and callers can set it to zero to recover about 688 MiB/GCD.
