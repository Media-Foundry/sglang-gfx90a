# DSpark gamma-3 compact M96 v2 rejection (2026-09-01)

## Motivation

Revisit confidence-pruned M96 after the accepted M96/M128 CK sparse-attention
kernels and M128 pre-router compaction. The prior M96 experiment masked routed
rows after routing; this v2 gathered the 32 dynamic anchor rows on device before
router and routed MoE, keeping routed compute at fixed M32.

## Graph correctness fix

PyTorch `index_copy_` with device-published dynamic row indices captured but
failed on the first heterogeneous BS32 replay with `hipErrorLaunchFailure`.
A unique-writer Triton gather/scatter replacement passed:

- 100 randomized BF16 inputs and row-index mutations;
- exact comparison with `index_select`/`index_copy_`;
- the INT64 one-dimensional hash-router token-ID path;
- 1000 HIP Graph replays, bitwise exact.

After that fix, 32 heterogeneous requests completed normally. This was an
experimental-path graph bug, not a hardware failure.

## Performance

Physical GCDs 4,5,6,7; TP4/EP1; original weights; gamma 3; compact mode;
M96 CK sparse attention enabled. A startup-only profiling override fixed the
draft budget to 2/3: 32 anchors plus 64 draft candidates, graph M96.

- BS1 France: historical exact prefix and semantic Paris.
- BS32 resident throughput: 946.86 and 1041.88 tok/s.
- Mean accepted length: 2.493 and 2.668.
- Host speculative step: 63.06 and 59.33 ms.
- All 32 requests produced 256 tokens with `finish=length` in both rounds.

The old SPS table was also tested without a forced budget. It over-pruned under
the much faster current service, yielding mean accepted length 2.13 and only
674.21 resident tok/s. The table's roughly 180-ms-era cost fit is stale.

## Decision

Reject M96 and remove the selector, gather/scatter kernels, and startup budget
override. Even a 59--63 ms step cannot compensate for the commit drop from the
accepted static M128 checkpoint (about 1.56k resident tok/s). A refreshed
adaptive SPS policy cannot exceed the best of these measured M96/M128 points;
future work must reduce M128 step time while retaining its accepted output.
