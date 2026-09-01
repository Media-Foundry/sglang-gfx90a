# DSpark M128 target-attention multi-stream checkpoint (2026-09-01)

## Motivation and current trace

A clean DSpark observer put the resident BS32 target-verify segment at
56.857 ms of a 69.533 ms GPU step.  A fresh layer-20 realtime-marker run on
the accepted checkpoint measured the stable M128 target layer at about
1.30 ms rank-max:

```text
MHC attention entry              ~97 us
attention prepare               ~279 us
attention core                 67--81 us
attention output                 ~179 us
FFN MHC entry                    ~112 us
MoE                              ~563 us
```

The prepare detail was serial:

```text
q_lora projection                 ~48 us
q norm                             ~13 us
wq_b                               ~53 us
fused Q/K norm+RoPE                ~16 us
indexer compressor/consumer        ~74 us
core compressor                    ~76 us
```

The CK sparse attention core is no longer the first attention bottleneck.
The missing overlap existed because the old HIP multi-stream path was decode
only: it wrote KV directly to the ring and returned no BF16 KV tensor.  Unified
TARGET_VERIFY must instead return the complete normalized candidate KV block
so the backend can store it before launching its per-row causal index streams.

## Implementation

Added `SGLANG_DSV4_GFX90A_DSPARK_TP4_M128_ATTN_MULTISTREAM`, default false in
the environment registry and default true only inside the explicit TP4-BS32
DSpark launcher profile.

The strict model-side selector requires all of:

- HIP/gfx90a path;
- TP4;
- C4 attention;
- exact physical M128 and global BS32;
- `ForwardMode.TARGET_VERIFY`;
- unified-KV backend;
- explicit switch.

Native AR cannot satisfy this predicate.  The path issues core and indexer
compressors on the existing auxiliary streams while the main stream computes
q_lora/q_norm/wq_b/QK normalization.  For target verify it disables the fused
kernel's own cache store, materializes and retains contiguous normalized BF16
KV, joins the compressor consumers, and returns `(q, kv)` to the unchanged
backend causal store.

## Correctness

- Candidate service captured all graph tiers without collective spin.
- Separate BS1 France oracle: 3/3 historical first-nine IDs exact and semantic
  Paris.
- All three real32 candidate rounds returned exactly 1024 tokens/request with
  `finish=length`.
- Frozen workload SHA256:
  `f74de67a93a660cde060991df71c9e2972a05d82c3ba3f9fe7c144b1f066a152`.

Concurrent France semantics can still follow a different long greedy path;
the separate BS1 exact oracle is the correctness gate, as in prior checkpoints.

## E2E ABBA result

Resident BS32 tok/s, 32 heterogeneous requests x 1024 output tokens:

```text
earlier independent control: 1572.30 / 1598.63 / 1572.51, median 1572.51
candidate:                  1646.59 / 1648.25 / 1621.66, median 1646.59
rollback control:           1576.08 / 1609.65 / 1594.04, median 1594.04
```

The midpoint of the two independent control medians is about 1583.3 tok/s;
the candidate is about +4.0%.  Candidate acceptance was
3.6035/3.5673/3.5541; rollback control was 3.5069/3.5970/3.5978, so the speedup
is not explained by a material acceptance increase.

Artifacts:

- `/tmp/dsv4_m128_attn_ms_france.json`
- `/tmp/dsv4_m128_attn_ms_bs32_r3.json`
- `/tmp/dsv4_m128_attn_ms_control_france.json`
- `/tmp/dsv4_m128_attn_ms_control_bs32_r3.json`
- `/tmp/dsv4_current_marker_lines.log`

## Decision

Keep the path in the TP4-BS32 DSpark profile.  It is below the usual 5% major
checkpoint threshold but passed the strict BS1 correctness gate and remained
positive between independent control services.  Do not enable it globally or
for native AR, legacy attention, other graph tiers, or other speculative widths.
