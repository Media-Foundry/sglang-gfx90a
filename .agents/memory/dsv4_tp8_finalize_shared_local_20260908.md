# TP8 M32 local routed/shared finalize re-screen

Baseline HEAD: fabfbda10b. This is an isolated GPU4 oracle, not a production
selector or a measured C1/C32 E2E improvement. Baseline service PID3325034 stays
unchanged with a 1,048,576-token KV pool. AMD-SMI PID audit found only its process
tree before testing. No weights, KV allocation, or communication protocol changed.

## Why revisit

The August 27 memory records an exact local finalize+shared-add oracle saving
1.42 us with existing AR, but its named script is absent from the current tree
and git history searched by that path. Do not revive its failed single-launch
peer-publication/AR variant. Current helper `maybe_fuse_routed_scale_and_shared_add`
still does a separate routed += shared for this path; its name does not imply
fusion into the down reducer.

## Contract and scope

Current production grouped reducer is reused as baseline. Candidate preserves
`((p0+p4)+(p1+p5))+p2+p3`, then BF16 rounding, then adds shared BF16 and rounds
to BF16 again. No atomics, peer access, or reordering of slots. Shape M32/T6/H4096.
Synthetic nonconstant inputs include varying exponent ranges and cancellation.
No claim of model-level correctness or cross-rank parity from these tests.

- 100 mutations: BF16 raw int16 representations match 100/100, max abs error 0.
- 1000 single-node graph replays: raw-bit exact and allocated memory unchanged.
- Original individual graph submission measurements: 7.5133 -> 6.1289 us.
- Independent raw-bit repeat: 7.6615 -> 7.5125 us, visibly noisy candidate.
- Final measurement captures 100 operations per graph to avoid Python submission
  gaps masking a microsecond saving. Five ABBA cycles, 10 samples/arm, trim one
  min and max: **4.071845 -> 2.475627 us**, saving **1.596217 us**.

Raw measurements are retained in the companion JSON, including the noisy run.
The burst is a warm repeated-address component test, not a full routed stage or
a system timeline. There is no new production throughput result.

## Integration constraint / next gate

Current dual-stream caller obtains fully reduced routed output inside experts,
then joins the shared stream, adds shared, and calls TP AR. Safe integration must
defer only the local reduction until the existing join, retain down work before
the join, and keep the original AR. Passing shared into the existing early expert
call would require an earlier join and could destroy overlap. Do not introduce
an implicit runner-global deferred tensor or leak state across batches.

43 times the local saving is about 68.6 us/step, roughly 0.22% of a 31-ms C32
step if entirely on the critical path. This is a budget, not an E2E prediction.
Candidate has no additional persistent workspace beyond existing partial and
output, but a future integration must validate graph-pool size and 1M KV capacity.
C1 uses a different reduction path and is not covered by this M32 candidate.

Files: `scripts/rocm/bench_dsv4_finalize_shared_local.py` and
`python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_finalize_shared_oracle.cuh`.
Both are standalone; no production import/selector was added.
