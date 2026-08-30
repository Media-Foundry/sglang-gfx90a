# DSV4 TP4 M32 attention-tail knockout oracle

Date: 2026-08-30

## Method

Four real gfx90a ranks use AIter's registered BF16 custom all-reduce.  The
input is formed from adjacent rank pairs in the real layer-20 TP8 M32 stage
dump, reshaped to the TP4 local attention-output shape `[32,2,4096]`.
Weights are deterministic BF16 tensors with production shapes.

The captured graph profiles are:

* A: inverse RoPE math + grouped `wo_a` + `wo_b` + all-reduce;
* B: capture-external exact `wo_a` output alias + `wo_b` + all-reduce;
* C: capture-external exact `wo_b` partial alias + all-reduce;
* D: exact reduced alias and an empty graph, measuring replay/launch floor.

No graph contains a copy that simulates removed work. Correctness passed 100
input mutations and 1000 graph replays, with exact intermediate, partial, and
reduced tensors. Timing uses rank maximum, A/B/C/D/D/C/B/A ordering, seven
rounds, 200 graph replays per sample, and a one-sample trim at each tail.

## Results

| profile | trimmed rank-max us |
|---|---:|
| A | 132.107 |
| B | 55.640 |
| C | 26.581 |
| D | 1.100 |

Gross removable upper bounds at this isolated boundary:

* inverse RoPE plus `wo_a`: **76.467 us/layer**;
* `wo_b`: **29.059 us/layer**;
* TP4 all-reduce: **25.481 us/layer**;
* total tail above empty-graph floor: **131.006 us/layer**.

## Interpretation

This is a knockout upper bound, not an additive production speedup forecast.
The inverse/`wo_a` profile uses Torch BF16 math rather than the production
wave64 grouped GEMV, so its 76.5 us is intentionally optimistic and must not
be quoted as the production component time. The `wo_b` and registered AR
segments use the production shapes and communication primitive and are more
direct lower-bound targets. The result says that removing AR alone cannot
close a large end-to-end gap; the compute tail must be attacked as a fused
producer-to-consumer boundary while preserving branch overlap.

Oracle: `scripts/rocm/bench_dsv4_tp4_m32_attention_tail_knockout.py`.
Raw log: `/tmp/dsv4_tp4_m32_attention_tail_knockout.log`.
