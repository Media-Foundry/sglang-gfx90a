# TP8 strict DSpark accept synchronization guard (2026-09-10)

## Trigger

During a refined-C4 control repeat, one of 32 real coding requests entered a
late two-token loop.  It generated 2048 tokens and its last-512-token duplicate
8-gram fraction was 0.871.  The first 128-token window above 0.6 duplication
started at generated token 1677.  Other same-profile rounds did not reproduce
the collapse, but full output hashes and even early greedy branches varied.

The generic `SYNC_TOKEN_IDS_ACROSS_TP=1` is not usable for DSpark: target and
draft workers do not enter the normal sampler collective in a compatible
order.  A direct test stalled after the first 16+16 request prefill and was
stopped; no production change uses that switch.

## Fix

`SGLANG_DSPARK_SYNC_ACCEPT_ACROSS_TP` makes rank 0 authoritative for the
target verifier's `(correct_len, bonus, cap_trim_lens)` tuple.  All three arrays
are packed into one contiguous int64 buffer and broadcast once per verify
step.  Every rank then rebuilds commit lengths, sequence lengths, and output
tokens from the same decision.

The guard also disables the folded in-graph accept/commit epilogue.  That is
required for correctness: folded commit writes target hidden KV before a
graph-external collective could synchronize the decision.  With the guard,
accept synchronization occurs before any target-hidden cache commit.

The switch is enabled only by the explicit strict TP8 full-target profile and
is otherwise default-off.  Native AR and non-DSpark execution do not call it.

## Validation

Real workload: TP8/EP1, gamma3, C32, original weights, 1M-token profile, 32
distinct coding prompts, greedy target, natural EOS/max2048.

Post-graph-only prototype (later superseded by pre-commit placement) completed
four waves at 1097.49 / 1085.96 / 1082.04 / 1096.41 tok/s.  The final
pre-commit implementation completed an excluded warm and measured wave at
1089.10 / 1077.26 tok/s.  Across all six waves (192 requests), no request
exceeded the 0.75 tail duplicate-8-gram gate; worst fractions in the first
four were 0.251 / 0.265 / 0.244 / 0.279.  The final two also passed the same
hard gate.  This is a quality/correctness hardening result, not proof of
cross-run bitwise determinism: output hashes still vary because rank-0
floating-point execution itself is not bitwise invariant.

The measured strict pre-commit rate is roughly 0.8--1.6% below the nearby
1095 tok/s control, within the cost expected for one tiny TP broadcast and
eager accept/commit.  Unit coverage checks contiguous packing and rebuilding
all derived commit outputs from the authoritative decision.  Two DSpark accept
tests pass.

Artifacts:

- `/tmp/dsv4_tp8_raw_vs_refined_c4_abba_20260910/A2_warm.json` (trigger)
- `/tmp/dsv4_tp8_sync_accept_quality_screen_v2_20260910/`
- `/tmp/dsv4_tp8_sync_accept_precommit_screen_20260910/`

