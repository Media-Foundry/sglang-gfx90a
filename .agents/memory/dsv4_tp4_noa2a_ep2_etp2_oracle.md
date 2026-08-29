# DeepSeek V4 TP4 M32 no-A2A EP2 x expert-TP2 oracle (2026-08-30)

## Question

Test the memory-conserving layout proposed by the external review without
Mori/token A2A:

- current A: each TP4 rank holds `E256/I512` and evaluates all top-6 routes;
- candidate B: two expert-owner groups hold `E128/I1024`; the two ranks in
  each group evaluate only that group's assignments, then a future global TP4
  reduction would combine the results.

This was a component-only oracle and was not wired into production.

## Workload and method

`scripts/rocm/bench_dsv4_tp4_m32_noa2a_ep2_oracle.py` reconstructs the exact
occupancy counts from the real 32-diverse-request recorder
`expert_distribution_recorder_1787803355.1849792.pt`, pass 37, learned-router
layer 34.  It keeps packed FP4, INT8 activation quantization, A4 metadata,
gate rows 2, down rows 2, eight waves, LDS LUT and the fixed-slot FP32/BF16
reduction.  Candidate masked-down includes the required partial-buffer clear.
Gate/down grids were independently swept over 416--2080 blocks.  Final timing
used `A/B0/B1/B1/B0/A`, five rounds and trimmed means on GCD0.

## Routing distribution

| layout | active experts | assignments | A4 scans | padding |
|---|---:|---:|---:|---:|
| A E256 | 106 | 192 | 113 | 260 |
| contiguous B0 | 53 | 107 | 58 | 125 |
| contiguous B1 | 53 | 85 | 55 | 135 |
| balanced B0 | 53 | 96 | 56 | 128 |
| balanced B1 | 53 | 96 | 57 | 132 |

The balanced split greedily minimizes A4 scans, then assignments, while
retaining exactly 128 expert IDs per group.  It is a generous oracle for any
static layout rather than relying only on contiguous IDs.

## Results

Contiguous owners, after per-group geometry selection:

- A: `439.596 us`;
- B0: `475.395 us` (`gate/down blocks=1664/2080`);
- B1: `426.624 us` (`2080/2080`);
- candidate rank-max: `475.395 us`, **7.53% slower** than A.

Balanced owners, with `2080/2080` selected for both groups:

- A: `439.265 us`;
- B0: `447.288 us`;
- B1: `455.264 us`;
- candidate rank-max: `455.264 us`, **3.51% slower** than A.

Balanced component medians were:

| profile | gate | quant | down | reduce |
|---|---:|---:|---:|---:|
| A | 255.892 us | 41.615 us | 171.501 us | 4.668 us |
| B0 | 264.208 us | 41.611 us | 167.821 us | 5.012 us |
| B1 | 267.168 us | 41.803 us | 173.285 us | 4.936 us |

The key reason is structural: halving A4 scans while doubling the expert
intermediate shard leaves approximately the same packed-weight bytes, while
the wider per-scan kernel and masked protocol do not reduce the critical path.

## Correctness and decision

This is explicitly a **performance lower bound**, not a mathematical output
oracle.  Synthetic weights cannot reproduce concatenation of four distinct
real TP weight shards, and the final inner-group/global collectives were
excluded.  Those omitted collectives can only make the candidate slower.

The candidate misses the required `>=10%` component gate even under a balanced
expert assignment and is already 3.51% slower before collectives.  Therefore
do not implement production no-A2A EP2 x expert-TP2 for TP4 M32.

