# TP4/M64 gate virtual-wave subgroup oracle: rejected

Date: 2026-08-30

## Question

After the full-wave `A1/R8 + A2/R4 + A4/R2` single-launch schedule was
rejected, test the genuinely different CDNA2 mapping in which one physical
wave64 is partitioned into occupancy-dependent virtual waves:

- A1: four independent 16-lane subgroups;
- A2: two independent 32-lane subgroups;
- A4: one ordinary 64-lane wave.

This was an isolated gate/up oracle only.  It was not wired into production.
It used the real heterogeneous TP4/M64 route recorder
`/tmp/expert_distribution_recorder_1788072257.651073.pt`, pass 20, layer 34,
and the production packed-FP4 weights, 1 KiB LDS E2M1 pair LUT, group-32 INT8
activations/scales, bounded SwiGLU, and G2080 reference geometry.

## ISA-safe mapping

The implementation did **not** treat a CDNA2 wave64 as a CUDA warp32 and did
not use DPP across virtual-wave boundaries.  Reduction was explicitly:

- width 16: `__shfl_down(..., 8/4/2/1, 16)`;
- width 32: `__shfl_down(..., 16/8/4/2/1, 32)`;
- width 64: `__shfl_down(..., 32/16/8/4/2/1, 64)`.

Thus a 16-lane reduction stays within one hardware DPP row, while the 32- and
64-lane cases use shuffle semantics with the declared subgroup width.

The subgroups did not mix experts.  All subgroups in one physical wave worked
on the same expert record but owned independent R2 row tiles:

- A1: four subgroups process four independent R2 tiles (eight output rows) for
  the same singleton assignment;
- A2: two subgroups process two independent R2 tiles (four output rows) for
  the same two assignments;
- A4: the full wave processes the normal R2 tile for four assignments.

This preserves weight/scale layout and avoids cross-expert divergence.  It is
also materially different from the rejected full-wave constant-product
oracle: A1/A2 K reductions actually use 16/32 lanes rather than scheduling a
complete wave64 per task.

## Route and work count

The real route contained 384 routed assignments and produced:

- 61 A1 records;
- 36 A2 records;
- 77 padded A4 records for all runs of occupancy at least three.

The physical wave-task count fell from 44,544 for production A4/R2 to 28,224,
a nominal 36.6% reduction.  Actual dot products and packed-weight bytes were
not reduced; fewer lanes performed proportionally more K groups.

## Correctness and graph stability

Across 100 in-place random activation/scale mutations, all four candidate
grids produced the same error envelope relative to the production DPP A4
reference:

- maximum BF16 absolute difference: `0.5`;
- maximum relative L2: `2.8419133741408587e-05`.

The difference is expected because A1/A2 use 16/32-lane reduction trees rather
than the established wave64 FP32 addition tree.  The candidate passed 1,000
HIP Graph replays with bitwise-stable output.

The source-level live accumulator budget was fixed at four assignments by two
rows for gate and up (16 FP32 values per lane in the generic runtime-mode
kernel), plus the 1,024-byte LDS LUT.  A separate HSACO resource continuation
audit was not run after every timed geometry missed the performance gate; no
resource claim is made beyond those explicit source allocations.

## Seven-round ABBA

Every candidate was tested in adjacent forward/reverse order for seven rounds
with 100 timed iterations per sample.  Tests used only physical GPU 6 after an
`amd-smi process` check; BIO processes were not stopped or modified.

| candidate grid | production trimmed mean | candidate trimmed mean | delta |
|---:|---:|---:|---:|
| 416 | 575.265 us | 624.765 us | **+8.605% slower** |
| 832 | 575.265 us | 620.287 us | **+7.826% slower** |
| 1040 | 575.265 us | 635.397 us | **+10.453% slower** |
| 2080 | 575.265 us | 654.260 us | **+13.732% slower** |

Full raw samples and correctness output are retained in
`/tmp/dsv4_tp4_m64_gate_subgroup_oracle.log`.

## Decision

Reject and delete the oracle implementation.  The best grid remained 7.83%
slower and was far above the predeclared `<=382 us` continuation threshold.
The task-count reduction is not a weight-work reduction: each narrow subgroup
still scans every K group for its row tile, but exposes fewer independent
lanes/waves to hide cold expert-weight latency.  Runtime mode control and the
narrower reduction trees add overhead, while the changed FP32 tree also loses
bitwise equality.  Do not productionize 16/32-lane virtual-wave packing for
this gate shape.  Future occupancy work must reduce actual weight scans/bytes
or fuse a useful consumer rather than only repack the same K work.
