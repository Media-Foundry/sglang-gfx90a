# TP4/M64 gate single-launch occupancy schedule: rejected

Date: 2026-08-30

## Question

Could one physical HIP launch replace separate A1/A2/A4 occupancy launches and
beat the production TP4 M64 A4 gate/up kernel while preserving its exact
arithmetic?  The isolated oracle used the real diverse-request recorder
`/tmp/expert_distribution_recorder_1788072257.651073.pt`, pass 20, layer 34.
It was never connected to a production selector.

## Route and schedule

The recorded TP4 route has 384 routed assignments, 146 active experts and a
maximum expert occupancy of 12.  Static CPU metadata produced:

- 61 singleton A1 records;
- 36 two-assignment A2 records;
- 77 A4 records for experts with occupancy at least three;
- 174 records total.

The final mapping kept a complete wave64 for every task and a constant product
of eight assignment/row accumulators:

- A1 uses one assignment and eight rows (`A1/R8`);
- A2 uses two assignments and four rows (`A2/R4`);
- A4 uses four assignments and two rows (`A4/R2`).

A runtime descriptor selected the mode inside one kernel launch.  All modes
retained packed FP4 weights, the 1 KiB LDS decode LUT, INT8 activation, the
production K traversal and SDOT sequence, and the exact wave64
`32/16/8/4/2/1` FP32 reduction tree.  The reference was the production
A4/R2/W8/LDS2 DPP gate kernel.

The production A4 schedule has `174 * 256 = 44,544` wave tasks.  Coarsening
A1 and A2 rows reduced the candidate to 28,224 tasks, a nominal 36.6% task
reduction.  It did not reduce the actual packed-weight bytes or dot products.

## Correctness and static resources

The final flat-accumulator implementation passed:

- initial BF16 output bitwise equality for grids 416/832/1040/2080;
- 100 in-place activation/scale mutations, bitwise exact after every mutation;
- 1,000 HIP Graph replays, bitwise stable.

HIP compiler resource analysis (`-Rpass-analysis=kernel-resource-usage`) for
the B416 candidate reported:

- 79 VGPR;
- 65 SGPR;
- 0 AGPR;
- 1,024 bytes LDS/workgroup;
- zero scratch, dynamic stack, VGPR spill and SGPR spill;
- compiler occupancy 6 waves/SIMD.

An earlier union-sized `[4][8]` implementation consumed 152 VGPR.  Flattening
the invariant eight accumulators removed that artifact and passed the declared
`<=96 VGPR` continuation gate.  A templated A1/R8, A2/R4, A4/R2 version used
104 VGPR and was not retained.

## Seven-round ABBA timing

Tests ran only on physical GPU 6 after an `amd-smi process` check.  BIO jobs on
physical GPUs 4/5 and heavy external CPU compilation caused noticeable clock
and timing drift, so the raw paired samples are retained below.  Despite that
noise, every candidate geometry lost by a wide margin.

### Grid 416

- production median: 570.022 us
- candidate median: 759.626 us
- delta: **+33.26% slower**
- A samples: `[492.116, 495.614, 535.851, 515.441, 533.621, 556.603, 579.541, 592.901, 560.504, 579.640, 609.778, 626.293, 625.022, 630.965]`
- B samples: `[669.736, 701.967, 724.968, 754.504, 696.616, 752.370, 769.487, 764.748, 790.418, 791.349, 745.839, 807.922, 827.365, 766.293]`

### Grid 832

- production median: 681.515 us
- candidate median: 808.120 us
- delta: **+18.58% slower**
- A samples: `[615.083, 712.197, 679.573, 671.720, 732.066, 712.866, 696.194, 624.901, 698.600, 735.378, 683.458, 447.630, 470.219, 492.737]`
- B samples: `[758.037, 815.925, 853.788, 718.728, 795.461, 804.649, 811.592, 896.626, 840.303, 984.643, 775.324, 836.844, 628.331, 643.880]`

### Grid 1040

- production median: 574.043 us
- candidate median: 728.199 us
- delta: **+26.85% slower**
- A samples: `[497.457, 515.205, 486.628, 576.577, 523.777, 571.509, 585.877, 556.021, 632.888, 651.470, 680.869, 580.049, 591.077, 568.430]`
- B samples: `[685.215, 685.343, 707.669, 663.307, 707.029, 665.976, 733.064, 723.333, 746.463, 741.429, 788.008, 808.581, 768.540, 847.845]`

### Grid 2080

- production median: 689.882 us
- candidate median: 770.176 us
- delta: **+11.64% slower**
- A samples: `[666.165, 703.327, 741.884, 691.291, 694.911, 688.472, 774.616, 721.147, 1009.305, 469.377, 477.384, 453.454, 493.345, 462.404]`
- B samples: `[784.700, 755.474, 755.653, 799.122, 835.593, 836.761, 1338.291, 799.560, 607.029, 1122.742, 580.145, 634.514, 609.285, 646.926]`

Full log: `/tmp/dsv4_tp4_m64_gate_single_launch_schedule_oracle_final.log`.
Resource log: `/tmp/dsv4_a124_final_resource.log`.

## Decision and explanation

Reject the single-launch A1/R8+A2/R4+A4/R2 schedule and do not wire it into
production.  The nominal task-count reduction is not a work reduction: each
coarsened wave still reads and decodes all independent weight rows and executes
the same SDOTs.  R8/R4 serializes more independent row streams inside one wave,
reduces the number of waves available to hide cold expert-weight latency, and
adds runtime descriptor/loop control.  Lower static VGPR count therefore did
not translate into better effective memory-level parallelism.  The result also
missed the predeclared `<=382 us` timing gate by a large margin.

Future occupancy work must reduce real weight scans/bytes or fuse a consumer;
merely coarsening row ownership while preserving all weight traffic is closed.
