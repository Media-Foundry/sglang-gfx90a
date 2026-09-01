# DSpark M128 split entry-MHC oracle (2026-09-01)

## Question

Could the first production integration of progressive M128 all-reduce advance
only the three draft rows' next-layer entry MHC while routed M32 and the late
anchor reduction complete?

The oracle calls the exact gfx90a production `mhc_fused_post_pre` backend and
preserves original request-major row mapping:

```text
[anchor, d0, d1, d2] x 32
draft compact rows: row % 4 != 0 (M96)
anchor compact rows: row % 4 == 0 (M32)
```

All GPU work used physical GCD 4 after `amd-smi` reported no processes.

## Correctness

The compact M96 and M32 outputs were compared against their corresponding rows
from one production M128 invocation for residual, post, comb and normalized
layer input:

- initial output: bitwise exact;
- 100 bounded rank-local state mutations: 0 failures;
- 1000 HIP graph replays: bitwise stable.

Thus entry MHC is row-separable for this split, but separability alone does not
make the schedule worthwhile.

## Seven-round symmetric timing

```text
production M128: 114.736 us
compact M96:      106.594 us
compact M32:       47.674 us
serial split:     154.268 us
hideable ceiling:  67.062 us/layer
```

The progressive all-reduce primitive costs roughly 28--29 us more than the
production M128 AIter collective. Therefore an MHC-only cross-layer pipeline
has a net engineering budget of only about 38 us/layer, below the pre-set
40--50 us continuation gate and likely only 2--3% E2E.

## Decision

- Do not connect an MHC-only pipeline to the production model.
- Retain the exact row-split result as a building block.
- Continue with the higher-upside strict-DSpark design: advance row-local,
  side-effect-free M96 attention producers after draft readiness; wait for the
  M32 anchor before the single full-block KV store/indexer/attention consumer.
- Never call full M96 `self_attn` independently because the candidate block's
  draft rows causally depend on the anchor KV and backend metadata is M128.

Oracle: `scripts/rocm/bench_dsv4_dspark_m128_split_entry_mhc.py`.
