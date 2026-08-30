# DSV4 TP4 M64 grouped-direct fallback audit

Date: 2026-08-30

## Effective selection under the TP4 BS32 profile

Although the profile name says BS32, its exported grouped-MoE variables remain
process-wide when a captured/eager M64 shape is encountered:

```text
assignments = 4
gate rows = 2
down rows = 2
gate blocks = 2080
down blocks = 832
LDS unpack = 1
M32 DPP/down-prefetch switch = 1
M32 logical-down-scale switch = 1
```

For an actual `[64,4096]`, Top-6 TP4 routed input, `aiter.py` resolves this as:

1. external per-token group-32 INT8 quantization;
2. AIter `moe_sorting(..., block_size=4)` (the fused quant/sort is strict M32);
3. generic grouped gate A4/R2/W8/G2080 with the 1-KiB LDS E2M1 LUT;
4. separate group-32 quantization of the gate intermediate;
5. generic grouped down A4/R2/W8/D832 with shuffled AIter scales;
6. the ordinary fixed-slot FP32 partial/reduction.

M64 is below the M1024 MFMA threshold, so no MFMA prefill path participates.

## M32-only guards and fallback safety

`use_m32_dpp_down_prefetch` requires every one of the following, including
`hidden_states.shape == (32,4096)`, Top-6 M32, exact TP4 weight shapes, A4/R2,
G2080, LDS mode, and non-MFMA execution. It is therefore false at M64.

`use_m32_logical_down_scale` is defined as the conjunction of that strict flag
and the logical-scale environment switch. Before down dispatch it is conjoined
again with `down_blocks == 832`. Consequently M64:

- never passes the logical `[E,N,K/32]` scale to a generic kernel;
- passes `quant_info.w2_scale`, the shuffled layout expected by generic down;
- never sets `use_row_prefetch` or `use_logical_scale` in the wrapper;
- cannot hit the strict M32 assertions in the specialized gate/down wrappers.

The load-time logical W2 copy is still allocated when the process-wide switch
is enabled (about 16 MiB/layer, 688 MiB/GCD for 43 layers), but M64 merely does
not consume it. This is a memory cost, not a correctness hazard.

Thus the M64 fallback is semantically safe and fail-closed. There is no path by
which a logical scale can silently reach the generic grouped kernel.

## Existing M64 geometry evidence

The only formal DSV4 M64 grouped-FP4 sweep is the TP8-shaped generic-kernel
surrogate, not the TP4 I512 shape:

| geometry | full routed median |
|---|---:|
| A4/R2/W8/B832 | 488.484 us |
| best A8: R1/W4/B1664 | 483.340 us |

The best A8 result was only 1.05% faster despite a theoretical 32.7--41.0%
reduction in weight scans. Other A8/R1 cases were about 490--505 us and A8/R2
was 519--527 us. All completed cases were exact within each assignment. The
predeclared 15% gate failed, so A8 was rejected.

This supports A4 as the robust M64 assignment choice, but it does **not** prove
G2080 is optimal for TP4 M64. The existing M64 A4 reference used B832, while
the current TP4 process inherits G2080 from the specialized M32 DPP gate. TP4
has I512 instead of the TP8 local I256 shape, so the old TP8 timings cannot be
used to select a new grid without a dedicated oracle.

## Decision

- Correctness: current M64 fallback is safe; logical-scale and DPP specialization
  are strictly unreachable.
- Assignment size: retain A4; existing A8 evidence is below materiality.
- Performance uncertainty: TP4 M64 gate G2080 is inherited rather than
  oracle-backed. If M64 becomes a target tier, the minimal new sweep is
  A4/R2/W8/LDS with gate blocks 832/1248/1664/2080 and down fixed D832, using a
  real diverse M64 route. Do not change production from the current audit
  alone.

