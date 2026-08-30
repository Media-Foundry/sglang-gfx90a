# DSV4 TP4 C4-indexer head-shard oracle (rejected)

Date: 2026-08-30

## Question

The C4 indexer currently replicates all 64 index heads on every TP4 rank.  This
oracle compared:

- A: each rank projects all 64x128 heads, computes all local logits and Top-512;
- B: each rank owns one contiguous 16-head weight slice, computes its local
  projection and additive ReLU-weighted score contribution, performs a real
  TP4 FP32 RCCL all-reduce of `[32,L]`, then runs the same Top-512.

This is an isolated synthetic-shape oracle, not a production selector.  It
uses the real M32, K1024, N8192, 64 heads, D128, C4 page-size 64 and L513/L640
shapes.  The q_lora, replicated wq_b and C4 cache values are deterministic
synthetic tensors because no complete captured indexer q_lora/wq_b pair was
available.  The logits kernel preserves the production structure-of-arrays
FP8 page layout and FP16-dot gfx90a configuration.

## Resource hygiene

`amd-smi process` was checked immediately before the accepted run.  GCDs 0--3
had no material resident process.  An earlier attempt overlapped another TP4
service and all of its timing output was discarded; only the clean rerun below
is reported.

## Correctness

The score decomposition is valid because each head contribution is additive
after per-head ReLU and weighting.

| C4 length | max abs score | relative L2 | exact Top-512 rows | mean/min overlap |
|---:|---:|---:|---:|---:|
| 513 | 0.007416 | 1.506e-6 | 32/32 | 512/512 |
| 640 | 0.006027 | 1.608e-6 | 32/32 | 512/512 |

The small FP32 score difference is the expected change from a 64-head local
reduction tree to four 16-head reductions followed by rank-order RCCL sum.  It
did not change any selected Top-512 set in this oracle.

## Clean four-rank rank-max ABBA

Seven forward/reverse rounds, five warmups and twenty iterations per sample:

| stage | L513 median/trim (us) | L640 median/trim (us) |
|---|---:|---:|
| full 64-head projection | 49.940 / 51.587 | 49.456 / 49.637 |
| local 16-head projection | 50.288 / 50.639 | 50.812 / 51.239 |
| full 64-head logits | 63.704 / 64.100 | 67.840 / 67.978 |
| local 16-head logits | 64.200 / 64.752 | 66.956 / 67.073 |
| TP4 FP32 score all-reduce | 272.184 / 272.865 | 262.184 / 262.488 |
| Top-512, A | 33.692 / 33.693 | 40.608 / 40.582 |
| Top-512, B | 33.748 / 33.740 | 40.560 / 40.567 |
| A full chain | 170.444 / 171.425 | 172.776 / 172.435 |
| B shard + all-reduce chain | 395.216 / 394.741 | 400.104 / 402.524 |

## Interpretation and decision

Reject the TP4 score-all-reduce design.

Head slicing does not accelerate either exposed producer:

- N2048 versus N8192 projection is effectively launch/weight-path limited at
  this M32 shape and remains about 50 us;
- the C4 logits grid is dominated by scanning/loading the same KV rows, so H16
  is essentially tied with H64.

Even before communication, the optimistic shard chain is only roughly
15--17 us below A.  A real FP32 RCCL score all-reduce costs 262--273 us and
makes B about 2.3x slower.  A future custom collective would need to be below
roughly 15 us merely to break even, while also preserving graph ordering;
existing TP4 collectives are not near that bound.  Do not integrate a C4
head-shard production selector.  The long-context bottleneck remains the
per-rank KV scan rather than replicated head arithmetic.

Reproducer:

`scripts/rocm/bench_dsv4_tp4_c4_indexer_head_shard_oracle.py`

