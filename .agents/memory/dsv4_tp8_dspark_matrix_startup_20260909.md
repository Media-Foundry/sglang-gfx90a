# TP8 DSpark matrix startup (2026-09-09)

The real-code TP8 AR matrix completed three measured rounds per C1/2/4/8/16/32.
Decode medians: 81.988, 104.341, 188.547, 329.125, 605.888, 1033.436 output tok/s.
8K-per-request prefill medians: 4724.712, 4987.075, 5233.403, 5061.248,
5163.334, 5253.259 input tok/s. C8/C16/C32 prefill first tokens were not
cross-round exact; do not label those as deterministic correctness passes.

The first DSpark startup inherited AR-only
`SGLANG_DSV4_GFX90A_TP8_M32_DOWN_FIXED_WARMUP=1` and failed in
`dsv4_down_fixed_warmup.validate_runner` before graph capture completed.
This was a configuration error, not OOM or a measured DSpark result. All eight
GPU processes exited and VRAM returned to idle. Do not remove the validator.

Retry explicitly disables fixed warmup and paired down graphs for DSpark only.
It uses TP8/EP1/no-A2A, original checkpoint, gamma=3, 1M token pool and graph
tiers 1/2/4/8/16/24/32. Dense-only graphs and M64/M128 anchor-only routed
approximations are disabled. Existing AR production defaults are untouched.

Retry service PID: 3961201; gate/controller PID: 3961202.
Log: `/tmp/dsv4_open_code_tp8_dspark_20260909_retry1.service.log`.
Controller log: `/tmp/dsv4_open_code_tp8_dspark_20260909_retry1.controller.log`.
Controller waits for the server ready marker, requires a Paris answer through
the chat API, then runs the six decode tiers with one warmup and three measured
rounds each. A France gate alone does not prove full numerical correctness.
No DSpark throughput result was available at the time this entry was written.
