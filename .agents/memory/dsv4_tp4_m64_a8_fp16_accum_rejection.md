# TP4 M64 A8 FP16-local-accumulator rejection

Date: 2026-08-30

The experiment kept original packed FP4 weights, group32 INT8 activations,
E8M0 scales, FP32 wave reduction, bounded SwiGLU and output BF16. Only the
long-lived per-lane gate/up partial accumulators were stored as FP16. The goal
was to reduce A8 VGPR pressure enough to benefit from fewer weight scans.

Real TP4 M64 pass20/layer34 routing:

```text
A4 blocks: 174
A8 blocks: 153
```

Ten activation mutations produced worst max-absolute error 2.4375, relative
L2 about 0.00151 and minimum cosine 0.9999988 against A4 FP32. Timing:

| rows | blocks | A4 FP32 | A8 FP16 | regression |
|---:|---:|---:|---:|---:|
| 1 | 416  | 413.241 us | 474.085 us | +14.72% |
| 1 | 832  | 413.340 us | 465.633 us | +12.65% |
| 1 | 2080 | 413.208 us | 454.141 us | +9.91% |
| 2 | 416  | 413.372 us | 449.708 us | +8.79% |
| 2 | 832  | 413.248 us | 444.896 us | +7.66% |
| 2 | 2080 | 413.184 us | 431.696 us | +4.48% |

The 12.1% scan reduction does not compensate for the wider assignment loop
and FP16 conversion/rounding work. Since the gate alone regresses and numerical
error is material, do not implement the down half or a production selector.

Reusable standalone oracle:
`scripts/rocm/bench_dsv4_tp4_m64_a8_fp16_accum_gate.py`.

