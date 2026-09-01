# DSpark M96/M32 split-boundary collective rejection (2026-09-01)

## Goal

Test the smallest dependency-correct form of the semantic-lane proposal at one
TP4 gamma-three boundary.  The candidate kept all rank collective calls in a
fixed order, reduced draft M96 before anchor M32, ran the draft row-local MHC
boundary on an alternate stream while the anchor collective executed, and
reassembled the original request-major M128 state.

Native AR was not modified.  All GPU diagnostics used physical GCDs 4--7.

## Correctness localization

The row-local MHC split itself is exact.  A single-GCD exact-shape check ran the
same M128 MHC boundary once as M128 and once as M96 plus M32; all four returned
states were bitwise equal after request-major reassembly.

The failure occurs at the collective boundary.  Comparing a registered AIter
TP4 M128 all-reduce with the M96/M32 row split on deterministic rank-distinct
BF16 inputs failed on the first mutation on every rank:

```text
anchor max abs: 0.125
draft  max abs: 11.0
```

Using three independent AIter communicator objects (one each for the M128
reference, M96 draft, and M32 anchor) did not change the mismatch, so this is
not merely the known back-to-back reuse of one communicator's peer temporary
buffer.  A constant-input standalone M96 test passed and measured about
85.57 us, demonstrating why constant all-reduce smoke tests are insufficient
for this shape; rank-distinct mutations are required.

The temporary boundary oracle was removed after the correctness gate failed.
No production selector, model-loop change, or AR behavior remains.

## 1-MiB geometry fine sweep

The accepted single-M128 collective was also swept at 9--15 CTAs on the same
four GCDs.  Slowest-rank graph medians were:

```text
blocks 9   50.821 us
blocks 10  48.432 us
blocks 11  47.998 us
blocks 12  47.651 us
blocks 13  47.938 us
blocks 14  48.585 us
blocks 15  48.310 us
```

This reconfirms the current `AITER_GFX90A_AR_1M_BLOCKS=12` profile as the local
optimum; no launcher change is warranted.

## Decision

- Do not implement semantic lanes as two ordinary M96/M32 all-reduces.
- Do not infer collective correctness from constant rank inputs.
- Preserve one logical M128 collective epoch and one reduction association.
- The next viable primitive is a single M128 peer-read reduction that publishes
  draft rows early (or performs their row-local epilogue internally) while
  retaining the existing M128 rank accumulation and final synchronization.

