# TP4 M64 no-A2A expert-owner lower-bound rejection (2026-08-30)

## Question

Would a memory-conserving no-A2A expert-owner decomposition improve the
DeepSeek-V4-Flash TP4/M64 routed stage?

* A: EP1 x expert-TP4, `E256/I512`, all routes on every rank.
* B2: EP2 x expert-TP2, `E128/I1024`, owned routes only.
* B4: EP4 x expert-TP1, `E64/I2048`, owned routes only.

Every rank already has the full M64 hidden state.  B does not dispatch or
combine tokens.  The candidate latency is the slowest expert-owner group and
the final global TP4 reduction is deliberately excluded, so this is favorable
to B.  It is not the previously tested Mori EP path.

## Inputs and fixed kernel stack

The route came directly from the heterogeneous-request service snapshot
`/tmp/dsv4_tp4_m64_real_route.pt`, layer 34/pass 20:

```text
M=64, topk=6
active experts=166
assignments=384
A4 weight scans=182
```

No synthetic count reconstruction was used.  Packed weight *values* were
synthetic but all tensor shapes and bytes matched the candidate layouts.
The A reference used the accepted component shapes:

```text
gate/up: A4/R2/W8, DPP reduction, G2080
down:    A4/R2/W4, row-prefetch, logical scales, D832
```

The owner partition greedily balanced `(A4 scans, assignments)` while keeping
equal expert counts.  It produced:

```text
EP2:
B0 active=83 assignments=193 scans=91 padding=171
B1 active=83 assignments=191 scans=91 padding=173

EP4:
B0 active=41 assignments=96 scans=46 padding=88
B1 active=42 assignments=96 scans=46 padding=88
B2 active=41 assignments=95 scans=45 padding=85
B3 active=42 assignments=97 scans=45 padding=83
```

Gate and down block counts were swept independently per owner.  Timing used
seven mirrored A/B owner-order rounds.  Before each GPU process,
`amd-smi process` was captured; tests used physical GCD 6 or 7 and no BIO
process was killed.

## Compact-owned activation accounting

Quantizing the full invalid-slot-padded `[M,T,I]` tensor would create a false
negative.  The final oracle instead timed both:

1. a preallocated `index_select` gather of valid owned `(token, slot)` rows;
2. INT8 group-32 quantization of the resulting `[owned_assignments,I]` tensor.

This measured about 46--51 us.  Candidate full time was calculated as:

```text
dense_full - dense_quant + (owned_gather + owned_quant)
```

The existing masked down protocol also requires a partial-buffer clear, which
was included.  The final collective remained excluded.

## Results

The host/GPU environment showed large absolute timing variance during this
window.  An A-only nine-sample repeat ranged from 732 to 1363 us full-stage
and from 471 to 968 us for gate, so the absolute numbers must not be presented
as a stable production baseline.  The decision therefore uses same-process,
interleaved A/B rank-max ratios and the generous 650-us absolute continuation
gate.

Two EP2 interleaved runs bracketed the result:

```text
run 1 before explicit gather accounting:
A full                         1005.13 us
EP2 owned-quant rank-max       1175.16 us
relative                       -14.47%

run 2 with gather + owned quant:
A full                         1232.16 us
EP2 corrected rank-max         1304.44 us
relative                        -5.54%

an earlier gather run, correcting A to dense full:
A full                         1034.11 us
EP2 corrected rank-max         1249.06 us
relative                       -17.21%
```

The formal EP4 gather+quant run was:

```text
A full                         1129.83 us
EP4 corrected rank-max         1242.35 us
relative                        -9.06%
```

The EP4 slowest group was B2 despite having only 45 scans and 95 assignments,
illustrating that scan balance does not remove the wider-shard kernel geometry
cost.  Neither candidate came remotely close to the continuation requirement:

```text
candidate rank-max <= 650 us AND >=10% faster than A
```

## Interpretation

Halving or quartering A4 scans does not halve weight traffic: expert-TP2 and
expert-TP1 respectively double and quadruple the local intermediate width.
The total packed weight bytes remain approximately conserved, while wider
shards worsen the current small-M gate/down execution geometry.  Owner
rank-max, compact gather/quant, partial clear, and the excluded final global
reduction add costs on top.  This is therefore not a promising production
route for the current gfx90a kernels.

## Decision

**Reject both EP2 x expert-TP2 and EP4 x expert-TP1 no-A2A layouts for M64.**

No production selector was changed and no correctness claim was made: the
oracle did not concatenate four real weight shards.  The temporary benchmark
implementation was removed after recording the failed experiment.

