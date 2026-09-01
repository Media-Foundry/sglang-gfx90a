# DSV4 raw-weight AIter prefill hybrid rejection (2026-09-02)

## Motivation

AIter's CK two-stage generator contains both preshuffled `a4w4` and
unshuffled `a4w4_bns` families. An experimental selector kept checkpoint-order
FP4 weights for the accepted direct C1/decode path and sent only large prefill
M to CK's unshuffled family. This avoided duplicating expert weights.

The two paths require different E8M0 scale layouts. The corrected experiment
therefore retained checkpoint-order raw scale caches while leaving the regular
shuffled scales for direct kernels. The extra cache cost was approximately
2.15 GiB/GCD over 43 routed layers; packed weights were not duplicated.

## Correctness and isolation

- The first attempt kept only raw scales and made the direct path answer France
  as `1`; this correctly exposed the scale-layout mismatch and was rejected.
- With dual scale layouts, France returned `The capital of France is Paris.`
  before and after raw-CK long-prefill requests.
- The selector required M above a configurable threshold. Small prefill and
  decode remained on the established direct implementation.
- Unit tests in `test_aiter_runner.py` remained 4/4 passing.

## Performance

For a real 4604-token `server_args.py` code-review prompt, raw CK steady TTFT
was 1.911--2.023 s, or approximately 2.28--2.41k input tok/s. This is slower
than the accepted direct M2304/M2300 plan at roughly 1.85 s / 2.49k tok/s, so
the threshold cannot include the C1 target shape.

For a real 13824-token prompt, the effective memory pool split execution into a
9984-token raw-CK chunk and a 3840-token direct tail. Warm complete-request
results were:

```text
2521.5 input tok/s
2495.6 input tok/s
2561.8 input tok/s
```

The raw CK branch is much better than the raw direct kernel at large M, but it
does not beat the approximately 3.1k preshuffled AIter service oracle and is far
from the 10k C32 objective. Its memory cost and extra layout contract are not
justified.

## Prefill-delayer defect exposed during testing

The five-millisecond cold-burst delayer can aggregate six or seven queued
requests when traffic is already arriving. However, on a cold solitary request
it can delay the first pass and then never receive a scheduler wakeup to check
the elapsed wall-clock timeout. Adding requests later also produced TP-rank
scheduler state divergence (three ranks CPU-spinning while one slept, GPU 0%).

Therefore the current queue-trigger delayer is not production-safe for combining
C1 and C32 profiles. A wall-clock field alone is insufficient: a delayed prefill
must install an explicit scheduler timer/wakeup, and every TP rank must consume
the same release epoch.

## Decision

- Remove the raw-AIter selector, raw scale caches, and tune-table changes.
- Keep the accepted raw/direct weights and established shuffled scale layout.
- Do not retry CK `a4w4_bns` without a kernel configuration that exceeds the
  preshuffled oracle and a memory-neutral scale-layout solution.
- Fix or replace cold-burst admission before using a large token budget as a
  C32 performance claim.

