# Strict DSpark same-expert wave-pod rejection (2026-09-01)

## Hypothesis

Strict gamma-3 verification spends most target passes at M64/M96 and repeatedly
scans the same expert in adjacent A4 chunks. The candidate grouped up to four
same-expert chunks in sibling waves of one CTA at the same output-row tile,
hoping ordinary gfx90a L0/L1/L2 behavior would reuse packed weight rows.

This differs from rejected LDS and persistent-row designs:

- every wave retains its existing A4 accumulators and reduction tree;
- no wave serially traverses a long expert run;
- no LDS payload exchange, CTA barrier, atomic queue, or host sync;
- the FP32 partial slot and final fixed-order reduction are unchanged.

The implementation is standalone-only and is not imported by a production
model or runner:

```text
python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_wave_pod_oracle.cuh
python/sglang/kernels/ops/moe/gfx90a_fp4_expert_wave_pod_oracle.py
scripts/rocm/bench_dsv4_dspark_strict_wave_pod_oracle.py
```

## Correctness

M64, M96 and M128 all passed:

- 100 activation, router-weight and route-metadata mutations;
- BF16 gate output, INT8 intermediate quantization, FP32 down partial and
  final BF16 output bitwise equality;
- 1000 HIP Graph replays with fixed graph addresses;
- Python compilation and gfx90a HIP JIT compilation.

All GPU work used physical GPU 4 after an `amd-smi` process check.

## Seven-round symmetric ABBA

| tier | repeat A4 chunks | production-style full stage | wave pod | change |
| ---: | ---: | ---: | ---: | ---: |
| M64 | 68.42% | 833.79 us | 975.19 us | **16.96% slower** |
| M96 | 26.36% | 1491.13 us | 2147.62 us | **44.03% slower** |
| M128 | 53.09% | 1757.34 us | 2200.85 us | **25.24% slower** |

Raw reports:

```text
/tmp/dsv4_strict_m64_wave_pod.json
/tmp/dsv4_strict_m96_wave_pod.json
/tmp/dsv4_strict_m128_wave_pod.json
```

## Decision

Reject the wave-pod structure and do not attach a service selector. Even M64's
high same-expert coverage cannot compensate for replacing the production
schedule's broad same-chunk row-tile parallelism with same-row multi-chunk
parallelism. The result misses both continuation gates (22% geomean gain and
270 us/layer saving) by a wide margin and should not receive a nearby grid
sweep.

The broader conclusion is that natural cache reuse is insufficient on this
kernel. Future full-routed work must preserve row-task parallelism while
reducing actual weight decode/read cost; merely co-locating equal-expert waves
is not enough.
