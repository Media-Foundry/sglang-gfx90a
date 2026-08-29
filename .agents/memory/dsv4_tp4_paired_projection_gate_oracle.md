# TP4 M32 projection-specialized paired-wave gate/up oracle (2026-08-30)

## Hypothesis

The production grouped FP4 gate/up kernel holds both A4 gate and A4 up
accumulators in each wave and emits a 96-VGPR code object.  Test whether two
projection-specialized waves can share one task: the even wave computes gate,
the odd wave computes up, and gate lane 0 hands eight FP32 results to the up
wave through 256 bytes of LDS.

## Implementation

The oracle uses a fixed one-task-per-wave-pair mapping so all 16 waves in the
1024-thread CTA encounter the cross-wave barrier uniformly.  Eight task pairs
per CTA require 3616 CTAs for the real diverse pass37/layer34 route (106 active
experts, 113 A4 scans).  Both waves retain the original group/K iteration,
packed FP4 LDS LUT, SDOT operation, scale multiplication and wave64 shuffle
tree.  Only the up wave applies bounded SwiGLU and writes BF16.

Files are standalone and are not connected to the production selector:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_gate_up_paired_oracle.cuh`
- `scripts/rocm/bench_dsv4_tp4_m32_paired_projection_oracle.py`

## Static resources and correctness

Extracted gfx90a HSACO metadata reports:

- 62 VGPR, 44 SGPR;
- 1280 bytes LDS;
- zero private/scratch allocation and zero VGPR/SGPR spills;
- maximum workgroup size 1024, wavefront size 64.

Thus the candidate reaches the requested `<=64 VGPR` tier and the 16-wave
workgroup is legal.  Against production A4/R2/W8/G2080:

- 100 independently mutated activation/router-weight iterations were exact at
  intermediate BF16, INT8 quantized value/scale, FP32 down partial and final
  BF16 output;
- 100 graph replays with mutated inputs were exact at all the same boundaries.

## Timing and decision

Seven-round ABBA, 30 iterations/sample on GCD0:

| stage | production A | paired B |
|---|---:|---:|
| gate/up | 255.073 us | 376.873 us |
| quant | 41.203 us | 42.016 us |
| down | 172.612 us | 172.635 us |
| reduction | 4.081 us | 3.871 us |
| full routed stage | 438.122 us | 559.115 us |

The full stage is **21.64% slower**, missing both the `<=395 us` and 10% gain
gates.  Reducing the grid to 2080 CTAs with two uniform rounds was also tested;
the required round-to-round barrier and invalid second-round pairs worsened the
full stage to about 720 us, so that variant was removed.

Although accumulator specialization achieves 62 VGPR, a 1024-thread CTA still
allows only one CTA per CU.  It doubles resident waves but does not add a
second CTA, while 3616 CTAs repeat LUT initialization and pay a full-CTA
gate/up exchange barrier.  Lower VGPR alone therefore does not improve this
memory-latency path.  Do not connect the paired-wave representation to
production.

