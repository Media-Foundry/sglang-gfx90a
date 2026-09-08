# Paired M32 graph: candidate warmup illegal address

After fixing the DSA dense key guard, reconstructed the service from current
`scripts/rocm_dsv4_flash.sh` plus recorded profile overrides. Original full
process environment was not preserved, so this is **not** an exact historical
environment reproduction. Public overrides are retained in
`/tmp/dsv4_tp8_paired_retry_20260908.json`.

PID 3502055: TP8/EP1/no-A2A, 1M KV, prefill36864, graph1/2/4/8/16/24/32,
DSA dual dense/sparse, row-stable, M32 overlap/legacy AR/gate-prefetch,
G832/D832/A4 confirmed from its live command/environment. All GPU PIDs were
absent before launching. It loaded weights but exited before readiness.

First Python exception on TP4 is in `capture_alternative -> _capture_one_impl
-> forward_fn` (candidate warmup), ending at multi-stream attention prepare
`q_norm -> x.contiguous()` with illegal device address. Other unwind errors
include failed pointer-attribute lookup during AIter IPC registration. The
trace is asynchronous: neither the reported attention call nor the subsequent
IPC error proves the original fault location. No paired-capture success log,
no memory-budget success, and no E2E correctness/timing result.

Controller 79352 exited with failed liveness assertion; process death and empty
amd-smi GPU inventory were verified before the next launch. Do not automatically
retry the same paired path. Keep the feature default-off.

## Immediate control experiment (live at this note)

Reconstructed baseline PID3507653 uses the **same explicit profile** with only
DOWN_PAIRED_GRAPHS=0. State:
`/tmp/dsv4_tp8_reconstructed_baseline_20260908.json`.
Service log uses the same stem plus `.service.log`.
Private mode0600 launch snapshot saved outside the repository; never publish it.

Controller session46881 waits for this PID (no automatic restart), then runs
C32 France and C1 two rounds/three real code tasks with the existing fixed
reference. C1 output:
`/tmp/dsv4_tp8_reconstructed_baseline_20260908.c1.json`.
France output has suffix `.france.json`. These were pending when this note was
written. Verify the live handle/state before further work. Compare complete
output IDs manually after the harness, which does not itself assert all
reference completion IDs.

If baseline capture succeeds, narrow the next component oracle to consecutive
full-model captures with auxiliary streams and reused forward/attention state;
the earlier small AIter dual-pool oracle did not exercise these interactions.
Do not generalize its 1000-replay result to the current full-model failure.
