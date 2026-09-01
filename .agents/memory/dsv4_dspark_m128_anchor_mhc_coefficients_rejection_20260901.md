# DSpark M128 anchor-MHC coefficient reuse rejection (2026-09-01)

The strict steady `TARGET_VERIFY` candidate computed MHC pre-mix and Sinkhorn
coefficients only for each request's anchor row, reused those coefficients for
the three draft rows, and retained each row's own residual weighted sum. The
padded early-exact BS33 tier, native AR, target weights and attention semantics
were unchanged.

Real heterogeneous 32x1024, original weights, stream interval 1:

```text
candidate: 1324.24 / 1425.96 / 1424.79 tok/s
median:    1424.79 tok/s
accept:    3.3070 / 3.3609 / 3.3457
France:    3/3 first-nine exact and semantic Paris
```

The adjacent exact-coefficient rollback centered near 1509 tok/s with accepted
length around 3.39--3.47. Although coefficient reuse reduced compute, it
degraded proposal acceptance enough to lose end-to-end throughput, including
the late time bins. Remove the tactic and do not use it as a correctness or
performance checkpoint.

Artifact: `/tmp/dsv4_anchor_mhc_candidate_bs32_1024_r3.json`.
