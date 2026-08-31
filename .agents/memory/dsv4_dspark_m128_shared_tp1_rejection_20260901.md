# DSpark gamma-3 M128 replicated shared-TP1 rejection (2026-09-01)

The existing `SGLANG_SHARED_EXPERT_TP1=1` path was tested as an exact-structure
oracle: every TP4 rank loads and computes a complete shared expert, then adds
that replicated result after the routed TP4 all-reduce.  This preserves all
M128 draft-row shared contributions and avoids summing replicated shared
outputs through the collective.  Original weights and TP4/EP1/no-A2A were
unchanged.

The service fit in memory and captured all graph tiers, leaving roughly
2.7 GiB/GCD during capture.  The first 32-real-request, 128-token round passed
semantic Paris but immediately regressed:

```text
resident throughput: 909.18 tok/s
mean accepted length: 2.3723
```

The contemporary TP4-sharded shared path is around 1.1--1.2k tok/s with mean
accept near 2.9--3.0.  Replicated TP1 therefore loses both compute time and the
accepted trajectory (different GEMM/reduction association).  Stop after the
screen; do not use shared TP1 to replace the M128 collective.

Artifact: `/tmp/dsv4_gamma3_shared_tp1_screen.json`.

