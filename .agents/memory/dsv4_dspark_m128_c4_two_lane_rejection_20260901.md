# DSpark gamma-3 M128 C4 two-lane scheduling rejection (2026-09-01)

## Candidate

The existing gfx90a C4 prepare path uses the main stream for the Q chain and
two side streams for the core compressor and indexer compressor.  A strict
default-off candidate reused one side stream for both compressor producers,
in production issue-order 3 (`core -> indexer`), while leaving the Q chain on
the main stream.  It required gfx90a, C4, TARGET_VERIFY, BS32, width four and
`x=[128,4096]`; native AR was unreachable.  No arithmetic, cache layout or
weight changed.

## Real-various service result

The first candidate service completed four accepted 256-token rounds:

```text
resident tok/s: 1187.79 / 1180.16 / 1169.93 / 1171.98
median:         1176.07
mean accept:    3.038 / 3.058 / 2.941 / 2.967
```

Its fifth France seam failed.  A subsequent eight-round, 128-token diagnostic
with mismatch recording passed semantic Paris in 6/8 rounds; failures diverged
at generated token 7 and used the recurring sequence beginning
`... 4987,1042,295`.  The candidate is pure scheduling, and the returned
control service also passed France only 3/5 rounds, confirming that this
particular semantic drift is an existing admission/batch-shape sensitivity,
not proof of changed C4 math.

The same-code returned control (ordinary three-lane prepare) measured:

```text
resident tok/s: 1185.69 / 1209.07 / 1124.21 / 1095.50 / 1155.40
median:         1155.40
trimmed mean:   1155.10
mean accept:    3.000 / 3.023 / 2.858 / 2.959 / 2.887
```

Although raw resident median favored the candidate by about 1.8%, the change
was explained by acceptance variance.  Resident throughput divided by mean
accepted length centered around 393 for the candidate versus about 395 for the
control; host/scheduler step metrics were also noisy and did not show a robust
latency reduction.  This is below the continuation threshold.

## Decision

- Remove the two-lane selector and retain the current C4 three-lane schedule.
- Do not interpret France seam drift as candidate correctness evidence in
  either direction; the first client request can decode before all 32 requests
  are admitted, so a BS32-only selector may not even be active at divergence.
- Future scheduling candidates need direct rank-max marker improvement plus
  acceptance-normalized E2E evidence, not raw resident throughput alone.

Artifacts:

```text
/tmp/dsv4_gamma3_c4_two_lane_5r.stdout
/tmp/dsv4_gamma3_c4_two_lane_mismatch8.json
/tmp/dsv4_gamma3_c4_two_lane_a2_5r.json
```

