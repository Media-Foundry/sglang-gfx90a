# DSpark gamma-3 M128 CK sparse-decode checkpoint (2026-09-01)

## Scope and guard

- Original DeepSeek-V4-Flash checkpoint.
- Physical GCDs 4,5,6,7; TP4/EP1/no-A2A.
- DSpark gamma 3; 32 heterogeneous requests, 1024 emitted tokens each.
- Fixed selected workload digest:
  `6699bf7e5153eaf6625f72954d8cad92064c944067874a6297bb780cf2958f5c`.
- Candidate selector:
  `SGLANG_DSV4_GFX90A_DSPARK_TP4_M128_CK_SPARSE_DECODE`.

The launch script enables the selector only inside its explicit DSpark + TP4
BS32 command block. Production dispatch additionally requires gfx90a, TP4,
BF16 KV, no fused inverse RoPE, exactly T=128/H=16/D=512, and contiguous
metadata. Native AR never receives the selector from the harness.

## Kernel extension and oracle

The existing native CK-style HIP kernel already sized its grids and workspace
from `args.tokens`; only `kMaxDecodeM`, the wrapper, and production selector
stopped at M96. Extending the bound to M128 did not alter kernel arithmetic.

On physical GCD 4, 100 random Q mutations and 1000 HIP Graph replays passed at
each context. CK output was graph-bitwise-stable. Maximum absolute error versus
the established Triton implementation was 0.0078125; maximum relative L2 was
0.003817.

| visible KV rows | Triton us | CK split2 us | saving us | CK speedup |
|---:|---:|---:|---:|---:|
| 128 | 129.907 | 77.117 | 52.790 | 68.46% |
| 256 | 195.558 | 119.462 | 76.096 | 63.70% |
| 512 | 312.535 | 182.144 | 130.390 | 71.59% |

The available split4 core was also tested and rejected. It measured
123.939/157.652/247.867 us for contexts 128/256/512, substantially slower than
split2 because extra core CTAs and reduction work outweighed the shorter scan.
No split4 production surface remains.

## Four-service A/B/B/A

Every arm used the same materialized prompt manifest and stream interval 1.
Control and candidate each contributed eight 32x1024 rounds across two
independent services.

| arm | service/round resident BS32 tok/s |
|---|---|
| A1 control | 1530.35, 1495.78, 1510.71, 1538.03, 1508.42 |
| B1 M128 CK | 1599.35, 1510.25, 1534.62, 1548.59, 1538.42 |
| B2 M128 CK | 1615.21, 1647.70, 1569.59 |
| A2 control | 1513.60, 1557.50, 1516.93 |

Combined medians:

```text
control A: 1515.26 tok/s
M128 CK B: 1559.09 tok/s
gain:      +2.89%
```

Control mean acceptance was about 3.57; candidate mean acceptance was about
3.58. Both arms passed concurrent France semantic 5/8, showing the same
pre-existing BS32 greedy trajectory variation. All four independent services
passed their separate BS1 France exact gate (A1 5/5, B1 5/5, B2 3/3, A2 3/3).
Every heterogeneous request completed the requested length.

The large standalone attention gain becomes only about 2.9% end-to-end,
indicating overlap/resource contention with the projection and compressor
branches. Nevertheless the gain is positive across the symmetric independent
service sequence and the correctness rate is unchanged, so the DSpark TP4
BS32 profile promotes the M128 split2 selector.

Artifacts:

- `/tmp/dsv4_m128ck_B1_1024_r5_allow.json`
- `/tmp/dsv4_m128ck_B2_1024_r3_allow.json`
- `/tmp/dsv4_m128ck_A2_1024_r3_allow.json`
- `/tmp/dsv4_m96hipb_control_A1_1024_r5_allow.json`
- `/tmp/dsv4_m128ck_{B1,B2,A2}_france_bs1_*.json`

This checkpoint raises the stable resident center from roughly 1.52k to about
1.56k tok/s. It is not completion of the 2k goal; the remaining gap still
requires target/draft structural overlap or another multi-millisecond target
verify reduction.
