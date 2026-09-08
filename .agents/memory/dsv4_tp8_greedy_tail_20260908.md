# TP8 greedy-tail oracle — 2026-09-08

Base1fdac7c0f8. Native TP8 service3277900 unchanged;1M KV pool preserved.
AMD-SMI audit before experiment found no foreign GPU processes. Independent
eight-rank RCCL oracle, small transient tensors only, original service idle.
No model/LM-head GEMM in this oracle; no production numerical change.

Baseline: gather eight BF16 [M,16160] vocabulary shards, reorder/cast FP32,
argmax. Candidate: local max/index, single all-gather of FP32 score/global ID,
select maximum score then minimum global ID. IDs below2^24 are exact.
This is greedy-only without penalties/logprobs/NaN handling; cannot replace
general sampling. Production gfx90a MultimemAllGatherer falls back to normal
TP gather, but this isolated default-process-group test is not a measurement
of the live service communicator, scheduling or graph tail.

Each run tested100 mutations per rank per shape, including ties, all-inf
(negative infinity), positive infinity: exact IDs throughout. Captured timing
five ABBA cycles,100 replays/sample, per-sample rank maximum then trimmed mean.

| Run | M | baseline us | candidate us |
|---|---:|---:|---:|
|initial|1|269.191|412.413|
|repeat|1|271.606|412.515|
|initial|32|446.763|418.720|
|repeat|32|442.651|426.034|

C1 regresses materially. C32 saves16.6–28.0us per step, NOT per layer:
roughly0.05–0.09% of the recent31ms C32 step if fully critical. This is an
engineering estimate, not E2E evidence. Do not integrate this generic candidate.

Initial process group teardown lingered after both results; exact oracle tree
3295402 was terminated, all tracked children exited. Added graph.reset before
destroy_process_group; independent full rerun completed exit0. Production
service was never restarted. No hardware failure inferred from teardown.

Script scripts/rocm/bench_dsv4_tp8_greedy_tail.py; raw timings adjacent JSON.
