# Row-stable prefill: debug-free E2E screen

Base commit7c4585ed22, native TP8/EP1/no-A2A, original weights, pool1048576,
graph tiers1/2/4/8/16/24/32. Candidate flag remains default-off.
Removed all six stage/attention dump environment keys before starting PID2977922.
Startup reported1048576 token capacity and15.64GB available/GCD.

C1 three-case medians83.8488/83.8598/83.4269 tok/s; six measured256-token
outputs exactly match the old reference. Six fixed-prefix next IDs also match,
but logprob arrays differ from the old GEMM. This is not an ABBA speed claim.
Raw `/tmp/dsv4_tp8_rowstable_e2e_c1_B_20260908.json`.

C32 same diverse code manifest, six rounds,256 output tokens/request:
E2E median738.4425 tok/s, resident1027.7553 tok/s. Round2 E2E979.8174,
but most rounds~738. The gap occurs before the resident decode window.
Cross-round exact full completions10/32; first divergence ranges8..194 for
the other requests. Fixed-prefix stability does NOT establish full AR stability.
Raw `/tmp/dsv4_tp8_rowstable_e2e_c32_B_20260908.json`.

Identified a candidate-specific compile risk: the fixed-order Triton projection
used M:constexpr. Changed M to a runtime argument with do_not_specialize=[M],
preserving fixed K order.10 M values5/17/63/64/65/127/128/369/736/1023 now share
one compiled artifact (hash456d837912d3d3af9be46ef8db3d81dd5f130616a613be71ea33acce6a8021e8)
and identical output rows.100 mutations and1000 graph replays passed on each
of the three existing test shapes. The checker now guards the artifact count.
E2E reduction of the waiting time is pending; do not yet claim this explains
the measured regression. No permanent weights/workspaces added.

## Runtime-M debug-free service

PID2987321, log `/tmp/dsv4_tp8_rowstable_runtime_m_B_20260908.log`.
Same config and inputs, only candidate M specialization changed.
Six-round C32 E2E median983.5179, trimmed983.3517, resident1031.5626 tok/s.
First round863.69 includes cold work; subsequent five median983.9921.
Thus recurring ~738 E2E behavior disappeared in this screen while resident
decode remained near its earlier value. This is a recovery of the experimental
candidate, not a proven gain over the accepted production~980 baseline; no ABBA
claim. Raw `/tmp/dsv4_tp8_rowstable_runtime_m_c32_20260908.json`.

Cross-round256-token exact7/32; differing requests first diverge8..249.
All192 requests produce256 tokens and finish length. This alone is not semantic
or bitwise correctness. Per-client exact counts differ with batch placement;
do not infer a random race solely from7 vs10. Remaining decode/tier/slot numerical
behavior still needs teacher-forced investigation and longer semantic tests.

Runtime-M C1 medians84.1989/84.2319/84.0744 tok/s; all6 measured full256-token
completions exactly match the established reference.
`/tmp/dsv4_tp8_rowstable_runtime_m_c1_20260908.json`.
France C32 again32/32 pass:
`/tmp/dsv4_tp8_rowstable_runtime_m_france_20260908.json`.
Current service has no stage dump flags, ROW_STABLE_PREFILL=1, same1M pool;
repository default remains0. Before promoting, run proper baseline/candidate
ABBA and finish the C32 decode stability investigation. Do not mix this
candidate's restored E2E speed with a production optimization gain.
