# TP8 M128 uniform-metadata and SDOT-interleave rejections (2026-09-10)

Two exact, component-only variants were tested on physical GPU 4 against the
accepted strict-DSpark TP8 M128 A4/R2/W8 row-prefetch routed path.  Neither was
connected to the model service.  Both passed 100 activation/scale mutations
and 1000 graph replays with bitwise-identical intermediate BF16, FP32 partial,
and final BF16 tensors.

## Down wave-uniform metadata

The subgroup-8 task geometry makes expert ID and four encoded token/slot IDs
uniform across a wave.  Applying `readfirstlane` had previously saved 3.6--5.7
us at native M32, so it was combined with the accepted M128 row-prefetch down
kernel.

| distribution | current row-prefetch | uniform down | result |
|---|---:|---:|---:|
| balanced | 753.88 us | 741.99 us | +1.60% |
| skewed | 706.50 us | 708.60 us | -0.30% |

The gain is distribution-dependent and below the 3% full-stage continuation
threshold.  Temporary source and benchmark wiring were removed.

## Assignment-interleaved SDOT

The second oracle kept A4/R2, weight loads, task mapping, per-assignment K32
order, FP32 accumulation order, and output addresses fixed.  It changed the
inner loop from eight dependent SDOTs for one assignment at a time to one SDOT
per each of four assignments before advancing to the next SDOT position.

Balanced full-stage ABBA:

| variant | latency |
|---|---:|
| current row-prefetch | 755.39 us |
| gate interleaved only | 1851.82 us |
| down interleaved only | 1199.55 us |
| both interleaved | 2302.79 us |

Skewed results were similarly negative: 706.18 / 1787.89 / 1222.11 / 2307.67
us.  Exposing independent instruction chains greatly extends live integer
accumulator/activation state; compiler scheduling and occupancy costs dominate.
Do not retry this loop interchange without an ISA-level scheme that avoids
simultaneously live assignment accumulators.

Raw logs:

- `/tmp/dsv4_tp8_m128_prefetch_uniform_balanced.log`
- `/tmp/dsv4_tp8_m128_prefetch_uniform_skewed.log`
- `/tmp/dsv4_tp8_m128_sdot_interleave_balanced.log`
- `/tmp/dsv4_tp8_m128_sdot_interleave_skewed.log`

