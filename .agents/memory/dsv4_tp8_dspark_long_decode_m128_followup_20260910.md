# TP8 DSpark long decode and M128 follow-up (2026-09-10)

## Long-window measurement

Configuration: strict full-target TP8/EP1, original weights, gamma=3, C32 graph
only, `stream_interval=16`, 32 heterogeneous code requests, two rounds, 1024
generated tokens per request. Metrics were enabled and the service was started
from a fresh process on all eight GCDs.

Results:

| round | group wall (s) | full-request aggregate tok/s | scheduler fallback tok/s | mean accepted length |
|---:|---:|---:|---:|---:|
| 0 | 188.291 | 174.03 | unavailable | 2.418 |
| 1 | 202.873 | 161.52 | 306.31 | 2.344 |

Both rounds passed the France first-nine-token and semantic-Paris checks. The
saved completion hashes were not identical across rounds
(`cross_round_all_exact=false`). The service had `max-total-tokens=8192`, so 32
requests at 1024 generated tokens were admitted in waves; the full-request
throughput is therefore not a resident C32 decode measurement. It is useful as
a long-output stability check, not as a peak-speed claim.

The result confirms that simply lengthening decode does not remove the
cross-round drift. The M128 down-consumer candidate must remain opt-in/off until
the first differing target state is isolated.

## M128 reducer status

The current grouped FP4 producer already writes a fixed-slot FP32 partial with
shape `[M, topk, hidden]`, and its reducer adds slots in a deterministic fixed
order before one BF16 cast. The standalone M128 oracle passed mutation and
1000 graph-replay bitwise checks, but the end-to-end persistent-workspace
candidate still drifted. Therefore the likely fault is outside the standalone
reducer (workspace/stream lifecycle, another target stage, or scheduler state),
not a reason to re-enable the candidate blindly.

Next safe step: retain production AIter stage-2, add a separately benchmarked
fixed-slot reducer oracle and compare per-layer/teacher-forced states before
attempting service wiring. Do not use BF16 CAS or claim the standalone speedup
as an E2E improvement.
