# DSpark global CKTile A16W4 split-K=2 rejection (2026-09-01)

## Question

Historical `SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT` failures were measured at
BS1/M1. Re-test whether DSpark gamma-3's compact target anchor M32 makes the
generic AIter/CKTile BF16-activation, FP4-weight path competitive.

## Configuration

- Physical GCDs 4,5,6,7; TP4/EP1; original checkpoint weights.
- DSpark block size 3 and the accepted TP4 BS32 profile.
- `SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE=0`.
- `SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT=2`.
- Fixed randomized heterogeneous manifest selected with seed 20260901.

This global switch affects both target and draft MoE tiers, so it is a service
screen rather than a clean target-M32 kernel oracle.

## Results

- BS1 France: 3/3 historical exact prefix and semantic Paris answer.
- Heterogeneous BS32, 256 tokens: resident throughput 720.94 and 712.97 tok/s.
- Mean accepted length remained about 2.91, so the throughput collapse is not
  explained by a loss of DSpark acceptance.
- Concurrent France was not semantic in either BS32 round, consistent with the
  already known concurrent greedy-trajectory variability.
- Accepted packed-SDOT plus M128 CK checkpoint center: about 1559 tok/s.

## Decision

Reject the global CKTile split-K path. It is approximately 54% slower at
service level. The generic path emits many untuned small-tier kernels during
graph capture and materially slows the draft as well as target execution.
Any future retry must be a target-verify-only M32 full-routed-stage oracle;
never re-enable the global selector for service testing first.
