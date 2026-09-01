# DSV4 prefill MFMA64 I32-owned gate/quant rejection (2026-09-02)

## Candidate

An oracle-only raw-weight kernel mapped one CTA to an A64 expert block and a
continuous I32 output group.  It sequentially evaluated two production-order
MFMA16 tiles, reused the 32-KiB split partial, stored BF16-rounded activation in
4 KiB LDS, and generated exact group32 INT8 values/scales in the producer.
The production selector was never changed.

Files retained as negative-oracle evidence:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_mfma64_gate_quant_oracle.cuh`
- `python/sglang/kernels/ops/moe/gfx90a_fp4_mfma64_gate_quant_oracle.py`
- `scripts/rocm/bench_dsv4_gfx90a_mfma64_gate_quant_oracle.py`

## Correctness

- M2048: 3 input/weight/scale mutations and 10 graph mutation replays exact.
- M2304: 1 mutation and 5 graph mutation replays exact.
- BF16 intermediate, INT8 values, FP32 scales and final routed BF16 output were
  bitwise equal to current MFMA64 gate + standalone quant + MFMA64 down.

These short gates were sufficient because performance missed the declared
threshold by a large margin; the planned 100/1000 run was intentionally not
spent on a rejected candidate.

## Performance

Physical GCD4, complete routed stage:

| shape | current (us) | I32 producer (us) | change |
|---:|---:|---:|---:|
| M2048 | 21062.780 | 26288.705 | +24.81% latency |
| M2304 | 23481.920 | 29902.776 | +27.34% latency |

Sequential I16 ownership and the longer CTA dependency/occupancy cost overwhelm
the removed quant launch and BF16 intermediate traffic.  Do not add a
production selector or repeat this design with geometry-only tuning.  A future
producer fusion would need simultaneous two-tile execution without doubling
live accumulator/LDS pressure, which is a materially different kernel.

