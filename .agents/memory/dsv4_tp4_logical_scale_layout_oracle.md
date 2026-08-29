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
