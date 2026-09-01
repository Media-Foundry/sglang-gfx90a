# DSV4 TP4 M4608 prefill rank-max profile

Date: 2026-09-02

The M4608 trace captured 16 real prefill forwards. Across 43 layers this gave
688 routed gate/up calls and 688 routed down calls.

Dominant median per-layer components were approximately:

- routed gate/up: 13.6 ms
- routed down: 10.4 ms
- routed core total: 24.0 ms/layer

The routed core therefore contributes roughly 1.03 s over 43 layers and is the
decisive high-concurrency prefill bottleneck. Collectives, MHC, sparse
attention, and indexer metadata are secondary at this shape.

An expert-major/N-tile/A64-block task reordering preserved exact output and
improved gate/up by 2.733%, while down changed by about 0.006%. The service
threshold is 5%, so the interleave was rejected and removed.

This result rules out further global task-order tuning as the primary path.
The next useful structural candidates are packed-FP4 variable-M grouped work,
active-expert tile unpack/reuse, and gate-epilogue quantization that removes a
full intermediate write/read without increasing occupancy pressure.
