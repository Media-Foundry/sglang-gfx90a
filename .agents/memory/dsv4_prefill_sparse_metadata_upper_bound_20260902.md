# DSV4 sparse-prefill top-k/SWA metadata upper bound (2026-09-02)

## Question

Would removing the materialized `combined_indices`/`combined_lens` tensor from
the exact sparse-prefill attention path materially improve the TP4/EP1 C32
prefill target?

## Measurement

On physical gfx90a GCD 4, the production Triton
`combine_topk_swa_indices` kernel was measured at the queue-aware M4608 shape:

- two real-request-shaped rows of 2304 query tokens each;
- top-k 512, SWA window 128, compression ratio 4;
- preallocated output buffers, matching the backend cache reuse path;
- output `combined_indices` size 11.25 MiB;
- 20 warmups, nine rounds of 200 launches.

```text
samples_us:
32.202 / 31.457 / 31.222 / 31.035 / 31.083 /
31.863 / 31.086 / 31.019 / 30.864

median:       31.086 us/layer
trimmed mean: 31.252 us/layer
```

Across 43 layers this is only about 1.34 ms per model forward.  A C32 round is
roughly 27 seconds, and even a single M4608 iteration is dominated by tens of
milliseconds of routed MoE.  Eliminating this kernel and all of its metadata
traffic cannot produce a visible fraction of the required 2.74k-to-10k gain.

## Decision

Do not change the sparse-attention ABI or fuse SWA index generation into the
attention kernel for the present short/medium-context C32 objective.  Retain it
as a possible long-context optimization, where indexer and attention costs grow
relative to routed MoE.  The immediate target remains repeated expert-weight
scans in the M4608 routed gate/down stages.
