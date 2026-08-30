# DSV4 TP4 M32 real-L20 wo_a FP16 oracle

Date: 2026-08-30

The oracle read layer-20 FP8 checkpoint weights/scales without modifying the
checkpoint, dequantized the TP4 rank-0 `wo_a` and `wo_b` shards to the same
logical BF16 reference, and used the real M32 attention-output dump.

Profiles:

* A: production BF16 grouped einsum;
* B: persistent FP16 weight, per-replay BF16-to-FP16 input cast, batched FP16
  matmul, BF16 output cast;
* C: pre-cast FP16 input ideal lower bound, otherwise identical to B.

All profiles were stable for 1000 graph replays. Across 100 mutations, B/C
had the same numerical difference:

| boundary | max abs | worst relative L2 |
|---|---:|---:|
| wo_a | 0.015625 | 0.00151293 |
| after production BF16 wo_b | 0.03125 | 0.00220291 |

Seven-round interleaved graph timing:

| profile | wo_a us | wo_a + wo_b us |
|---|---:|---:|
| A BF16 | 39.256 | 74.156 |
| B FP16 including input cast | 53.986 | 91.019 |
| C pre-cast FP16 ideal | 43.154 | 80.379 |

The real candidate is 27.28% slower for wo_a and 18.53% slower through wo_b.
Even the pre-cast ideal is 9.03% slower for wo_a and 7.74% slower through
wo_b. Neither approaches the required 32.7 us threshold, and the numerical
error grows through the production consumer.

Decision: reject FP16 wo_a on the current ROCm matmul path. Do not run a
service A/B and do not wire production. The result reinforces that CDNA2 FP16
does not automatically outperform the tuned BF16/rocBLAS M32 GEMM merely by
changing operand dtype.

Oracle: `scripts/rocm/bench_dsv4_tp4_m32_woa_fp16_oracle.py`.
Raw log: `/tmp/dsv4_tp4_m32_woa_fp16_oracle.log`.
