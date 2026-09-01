# DSV4 M4608 expert-persistent MFMA64 rejection (2026-09-02)

## Hypothesis

For the queue-aware TP4 prefill tier, two 2304-token requests form an exact
M4608 routed-MoE batch.  With Top-6 and 256 experts, a balanced route gives
about 108 assignments per expert, normally represented by two consecutive
A64 sorter blocks.  The candidate maps one CTA to an expert/output tile and
walks all of that expert's A64 blocks before leaving the weight tile, instead
of scheduling each block independently.

The implementation preserves the production MFMA accumulation and fixed-slot
down-reduction order.  A GPU histogram/run builder derives active experts,
block starts and block counts without a device-to-host synchronization.

## Standalone oracle

Hardware: physical gfx90a GCD 4.  Shape: E256, M4608, Top-6, H4096, I512,
raw original FP4 weights and group-32 INT8 activations.

Gate/up, 20 route/input mutations and seven-round ABBA:

```text
production MFMA64:       14538.562 us
expert-persistent:       13506.713 us
speedup:                     7.640%
bitwise exact:             20 / 20
```

Down, including the production fixed-slot reduction in the control:

```text
production MFMA64+reduce: 14022.741 us
expert-persistent partial:12981.281 us
reduction:                    about 5 us
effective speedup:            about 8.0%
bitwise exact partial:       20 / 20
```

The combined routed-core saving is about 2.07 ms out of 28.56 ms, or 7.3%.

## Service ABBA

Configuration:

- original DeepSeek-V4-Flash weights
- physical GCDs 4--7
- TP4 / EP1 / no A2A / native AR
- queue-aware 2304/4608 chunk policy
- token-row-owner MHC prefill
- 32 distinct real code prompts, 73,724 audited input tokens
- one generated token per request

Results:

```text
A1 control:   2654 / 2801 / 2780 / 2724 input tok/s
              trimmed center 2752

B candidate:  1812 cold, then 2497 / 2548 / 2508 input tok/s
              warm center 2502

A2 control:   2678 / 2753 / 2764 input tok/s
              warm center 2753
```

The candidate regresses warm service throughput by about 9.1%.  Per-forward
server markers similarly fell from roughly 2.88--2.90k to 2.50--2.62k input
tok/s.  The most likely explanation is lower service-level occupancy and a
longer rank tail: retaining one expert/output tile per CTA improves cache
locality in isolation but constrains independent block scheduling across the
full model and four ranks.

France remained semantically correct:

```text
The capital of France is Paris.
```

The existing heterogeneous prefill harness does not have cross-round first
token bitwise stability even in both control arms, so this fragile signal was
not attributed to the candidate.  The kernel-level mutation oracle remained
bitwise exact.

## Decision

Reject production integration.  Keep the standalone header and benchmark
scripts as an exact oracle, but leave the service selector disconnected.  Do
not retry a fixed expert-persistent work decomposition unless a future design
can retain global block-level scheduling (for example a low-overhead static
work queue or a selector limited to demonstrably high-occupancy experts).
