# TP4 strict DSpark gamma-3 verify-budget sweep (2026-09-01)

## Question

Can compact M64/M96/M128 target graphs recover the approximate anchor-only
throughput while retaining strict target-verification semantics?

## Configuration

- Physical GCDs: 4,5,6,7.
- DeepSeek-V4-Flash original weights, TP4 / EP1 / no A2A.
- DSpark block size 3; folded draft CUDA Graph enabled.
- Full routed MoE on every real target-verify row:
  `SGLANG_DSV4_GFX90A_DSPARK_M64_ANCHOR_ONLY_ROUTED=0` and
  `SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED=0`.
- Captured compact token tiers: 64, 96, 128.
- 32 distinct code/chat requests, 256 generated tokens, stream interval 1.
- Two rounds per forced verify-budget fraction.

## Results

| forced budget | resident median tok/s | mean accepted length | France first-nine |
| ---: | ---: | ---: | --- |
| 0.30 | 723.19 | 2.15 | 2/2 exact |
| 0.40 | 738.88 | 2.23 | 2/2 exact |
| 0.50 | 709.45 | 2.23 | 2/2 exact |
| 0.60 | **742.20** | 2.26 | 2/2 exact |
| 0.80 | 684.33 | 2.43 | 2/2 exact |

All requests in every round produced exactly 256 tokens with
`finish_reason=length`.

## Conclusion

The sweep is correctness-clean but negative for the throughput goal. Raising
the budget from 0.6 to 0.8 increases accepted length by only about 7%, while
the larger full-target tier reduces resident throughput by about 8%. Lower
budgets do not recover enough model time to offset the lower accepted output.

This establishes that the prior roughly 1.0--1.1k gamma-3/5 results came from
the non-exact anchor-only target approximation rather than verify-budget
tuning alone. Further strict DSpark work needs a faster full routed-MoE target
kernel or a different exact verification decomposition; more scalar budget
sweeps are low value.

Evidence:

```text
/tmp/dsv4_gamma3_exact_f030_r2.json
/tmp/dsv4_gamma3_exact_f040_r2.json
/tmp/dsv4_gamma3_exact_f050_r2.json
/tmp/dsv4_gamma3_exact_f060_r2.json
/tmp/dsv4_gamma3_exact_f080_r2.json
```
