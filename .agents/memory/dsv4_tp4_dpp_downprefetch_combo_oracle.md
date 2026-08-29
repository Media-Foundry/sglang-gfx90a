# TP4 M32 DPP-gate plus down-row-prefetch combination oracle (2026-08-30)

## Profiles

This final exact-stack oracle used the real diverse pass37/layer34 route
(106 active experts, 113 A4 scans) and compared:

- A: production shuffle gate W8/G2080 plus production down D832;
- B: gate-only DPP W8/G2080 plus exact down row-prefetch D832;
- C: gate-only DPP W4/G2080 plus the same down row-prefetch.

All profiles retain A4/R2, packed FP4, LDS LUT, INT8 quantization and the
fixed down reduction.  The benchmark is
`scripts/rocm/bench_dsv4_tp4_m32_dpp_downprefetch_combo_oracle.py`; it reuses
the existing DPP and row-prefetch oracle kernels and is not wired to
production.

## Correctness

Across 100 mutated activation/router-weight inputs, B and C matched A bitwise
at intermediate BF16, quantized INT8 value/scale, down FP32 partial and final
BF16 output.

## Seven-round A/B/C/C/B/A result

Trimmed means, 30 iterations/sample on GCD0:

| stage | A | B: DPP W8 + prefetch | C: DPP W4 + prefetch |
|---|---:|---:|---:|
| gate | 257.764 us | 246.360 us | 246.331 us |
| quant | 40.798 us | 40.158 us | 39.756 us |
| down | 172.696 us | 170.523 us | 170.313 us |
| reduce | 4.957 us | 4.416 us | 4.331 us |
| full routed | 441.501 us | 426.644 us | 428.796 us |

B improves the component by 3.48%, while C improves it by 2.96%.  W4 does not
accelerate the isolated DPP gate and makes the composed full path roughly
2.15 us slower than W8.

The predeclared hard gates were B `<=421 us` and C `<=416 us` with at least 5%
gain.  Neither passes.  In particular, the best B remains 5.64 us above its
continuation threshold and below the required 5% component gain.  Per the
test contract, record and archive this exact combination without starting a
service or adding production selectors.

