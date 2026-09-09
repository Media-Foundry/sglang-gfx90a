# TP8 strict DSpark Q2xH8 C128 pair oracle (2026-09-10)

## Scope

The experiment packed two adjacent target-verification queries with eight
local heads each into the otherwise half-empty M16 dimension of the existing
gfx90a BF16 MFMA sparse-attention kernel.  It retained independent softmax,
sink, split workspace, and output rows.  It was restricted to strict TP8,
gamma-three, full-target M128 C128; C4 was never selected.

## Component evidence

A real eager fixture was captured from rank 0, layer 3, using 32 heterogeneous
code prompts.  All 64 adjacent pairs had an ordered-prefix physical index
contract and equal 16-key tile counts.  Against the current production H8
kernel, the production-arithmetic pair oracle was bitwise exact.  Seven-round
graph ABBA measured:

```text
production H8 median: 37.2664 us
Q2xH8 pair median:    31.5030 us
gain:                 18.2946%
saving:                5.7634 us/layer invocation
```

The earlier synthetic length-21--30 fixture measured a larger 39.1% gain;
the real fixture is authoritative.  A separate refined-probability version
was numerically valid but intentionally not used for exact production wiring.

## Service rejection

An A/B/B/A service run was started with fresh services, strict full-target
verification, synchronized draft/accept decisions, the accepted draft-M96
row-prefetch path, and 32 real code requests.  A1 control measured:

```text
warm: 1086.8757 resident output tok/s
test: 1083.1589 resident output tok/s
severe repetition: none
```

B1 measured:

```text
warm: 805.6597 resident output tok/s
test: 811.5859 resident output tok/s
severe repetition: request 2 tail-8 duplicate ratio 0.8851
```

The run stopped at the correctness gate; B2/A2 were not executed.  The
production selector, environment variable, and backend wiring were removed
immediately.  No unsafe path remains enabled.

The layer-3 fixture proves only one layer and one early step.  Longer decode
can cross a 16-key split boundary, where adjacent rows can have different
tile counts; other steps may also violate physical-prefix assumptions.  The
current packed kernel follows the longer row's split schedule, so it is not
an exact fallback for those cases.  A future attempt must preserve each row's
independent split partition and use a device-side fallback for incompatible
tiles before any service selector is restored.

Artifacts:

- `/tmp/dsv4_q2h8_real_fixture_20260910_l3/layer_3_rank_0_c128.pt`
- `/tmp/dsv4_tp8_q2h8_abba_retry1_20260910/`
- synthetic/component command output was retained in the agent transcript.

Decision: keep only the offline oracle and replay support as evidence.  Do
not enable Q2xH8 in production until the per-pair fallback is exact across
tile-count transitions and full E2E correctness passes.
