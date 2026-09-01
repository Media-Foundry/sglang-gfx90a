# DSpark M128 progressive all-reduce oracle (2026-09-01)

## Scope

This is a standalone TP4/gfx90a communication oracle.  It does not change the
production model, native AR, model weights, or the accepted DSpark profile.
All GPU runs used physical GCDs 4--7 after `amd-smi` reported no processes.

The goal is one logical M128 collective epoch which preserves the production
AIter two-stage numerical association while making gamma-three draft rows
available before anchor rows:

```text
request rows: [anchor, d0, d1, d2] x 32
draft phase:  original global rows with row % 4 != 0
anchor phase: original global rows with row % 4 == 0
owner:        global_row // 32
sum order:    owner, owner+1, owner+2, owner+3 (mod 4)
```

The implementation uses AIter direct HIP allocations and registered peer
pointers.  One rank-global entry epoch surrounds a compact 9-CTA draft grid
and a compact 3-CTA anchor grid, followed by one rank-global exit epoch.  A
single-wave side-stream gate arms the epoch and waits for all nine draft CTAs
to publish completion.  Every peer sum accumulates in FP32 in the owner-rotated
order and downcasts to BF16 once.

## Correctness

Compared with the production AIter 12-block 1-MiB all-reduce:

- 100 rank-distinct activation mutations: full M128 and compact M96 snapshot
  bitwise exact on every rank;
- 1000 HIP graph replays: zero mismatches on every rank;
- maximum absolute difference: 0;
- no stale epoch, spin, or device fault.

## Four-rank ABBA

Seven rounds, 100 replay/leg, rank-max medians:

```text
production AIter M128 AR:          64.649 us
progressive AR only:               93.071 us (representative center)
progressive + M96 draft snapshot: 108.121 us
eager draft-ready event median:    96.400 us
eager full-progressive median:    130.000 us
```

The direct peer-read progressive primitive is roughly 28--29 us slower than
the production two-stage collective when measured alone.  It is not a drop-in
AR speedup.  Its value is that the draft consumer can start before anchor
completion; the previous dependency-correct compute-only semantic-lane oracle
hid 191.740 us/layer.  Subtracting the extra communication still leaves enough
theoretical budget to clear the 100-us/layer service continuation gate, but
this must be proven by a real boundary composition.

A 6-CTA draft grid was also screened.  Progressive-only remained about 95 us
and eager draft-ready worsened to about 142 us, so the 9-CTA draft / 3-CTA
anchor geometry is retained.

## Decision

- Keep the standalone oracle and proceed to a real shared-M96 / routed-M32
  boundary composition.
- Do not replace production AIter all-reduce with this primitive by itself.
- Stop before service integration unless the real four-rank boundary saves at
  least 100 us/layer and routed-M32 contention stays below 5%.

Implementation:

- `gfx90a_tp4_m128_progressive_ar_oracle.cuh`
- `gfx90a_tp4_m128_progressive_ar_oracle.py`
- `bench_dsv4_tp4_m128_progressive_ar_oracle.py`
