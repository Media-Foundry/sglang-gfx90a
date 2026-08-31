# DSpark gamma-three confidence-compact M96 rejection (2026-09-01)

## Scope

- Original DeepSeek-V4-Flash checkpoint, TP4/EP1, physical GCDs 4,5,6,7.
- 32 concrete heterogeneous coding requests, greedy, 256 output tokens.
- Gamma three with compact ragged verification and a forced confidence budget
  of 2/3. The scheduler admitted 64 of 96 draft candidates plus 32 mandatory
  anchors and aligned the target graph to the M96 tier.
- A temporary strict selector kept routed MoE only on the 32 request anchors.
  It required gfx90a, TARGET_VERIFY, BS32, gamma-three width four, a compact
  layout-provided anchor mask, exact M64/M96/M128 shapes, and an explicit env.
  Native AR could not match the selector.

## Correctness gate

- France first nine token IDs: exact in all three rounds.
- France semantic answer: Paris in all three rounds.
- 32/32 requests in every round generated exactly 256 tokens with
  `finish=length`.
- Cross-round completion hashes were not stable; as with the existing
  speculative approximation, scheduling/bonus trajectories differed.

## Performance

Resident BS32 throughput:

```text
870.348, 828.438, 842.592 tok/s
median 842.592 tok/s
```

Aggregate request wall-time throughput was 492.506/496.312/499.373 tok/s.
Mean accepted lengths were 2.335/2.265/2.231. The accepted static gamma-three
M128 checkpoint is around 0.90k resident tok/s and typically retains a higher
accepted length. Removing 32 verify rows did not recover the lost output value.

## Decision

Rejected. The temporary RaggedVerifyLayout anchor-mask field, graph staging,
model selector, environment variable, and tests were fully removed. Keep the
static gamma-three M128 anchor-only routed checkpoint. No AR code/default was
changed.

Raw report: `/tmp/dsv4_gamma3_compact_m96_anchor.json`.
