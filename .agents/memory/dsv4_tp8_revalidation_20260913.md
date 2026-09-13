# Original V4 Flash TP8 revalidation and P/D matrix

User explicitly requested latest-main regression at a few sizes, followed by
the M128 down-consumer experiment, and then TP8 C1/2/4/8/16/32/64 P/D.
Decode AR and strict full-target DSpark are separate arms. Original weights
and1M logical pool are retained. Candidate acceptance requires correctness
and no demonstrated E2E slowdown; optimize via ABBA, final matrix three rounds.

**Scope update: user subsequently cancelled DSpark measurements.** Proceed
only with TP8 native-AR C1/2/4/8/16/32/64 P/D. The DSpark M128 experiment is
deferred, not rejected by a new measurement. Its untested standalone harness
extensions and DSpark launcher override were removed after the scope update.
No candidate production selector was added. Historical1127DSpark is not an
AR comparison point.

Starting HEAD `a365956778`; backup branch
`backup/pre-v4-tp8-regression-20260913`. V4.1 stays frozen; its service is off.
Before starting, `amd-smi process` showed no processes on GPU0--7.
Unrelated dirty files are preserved, including the graph-memory pickle.

## Plan and measurement contracts

1. Fresh strict gamma3 process on current code, diagnostic tiers1/8/32,
   original `/home/pc/models/modelscope`, pool1048576, chunk2304. Confirm
   Paris, readable real code, actual graph/kernel selection, and short pilot.
2. Revisit M128 down-consumer only after a usable control. Preserve full
   target experts and draft/accept TP synchronization. Component exactness
   alone and cross-wave generated hashes alone are insufficient verdicts.
3. Final C1/2/4/8/16/32/64 matrix with explicit graph tiers and actual row
   counts. P uses8K code input and1 output token; D uses real code and natural
   EOS with bounded output length. Measure common resident windows separately
   from admission/drain/HTTP wall time. Keep inputs fixed across matched arms.
4. Persist inputs, readable outputs, hashes, timestamps, launch configuration
   and summaries outside `/tmp`; prior raw matrix artifacts were lost.

Historical references: AR C32 matrix1033.44resident tok/s; short256-output
AR geometry ABBA1050.80resident/1002.61HTTP tok/s; final strict DSpark MHC
FP32 family1127.48resident mean on512-output ignore_eos protocol. These are
not identical workloads. Old TP4 approximate target1.5k is not a strict goal.

No new performance or correctness result at this initial record.

## Fresh-start regression, first attempt

The first fresh process (PID487458) loaded target and bundled draft weights,
then failed before ready: unified KV initialization registered the upstream
`clear_c4_req_states` callback, but the merged pool implementation lacked it.
No request or speed measurement ran. The failed log is retained separately.

The same merge gap left unified BF16 KV charged with paged-SWA/C4 linear
storage, silently reducing the requested1M pool to745728. Restored request-ring
C4 sizing/clear callback and separated fixed unified SWA/C4 state costs from
per-token compressed KV costs. Retained this fork's actual speculative ring
size and preserved the non-unified V4.1 formula; no attention math changed.

Validation before second launch: C4 lifecycle/budget tests9passed;
`scripts/rocm/check_dsv41_unit.sh`128passed plus16subtests;
new benchmark harness CPU tests3passed. The broader pool-configurator test
file could not collect because the environment lacks `datasets`; it is not
claimed passing. GPU process scan again showed all eight devices idle.
Second fresh startup and E2E are pending.

Second process499788 confirmed full=1048576 on every rank, coefficient6373.86
bytes/token, fixedC1285.05GiB and unified rings2.24GiB; after target/draft
pools about19.7GiB remained per GCD. It then failed at target graph metadata:
V4.1 ratio discovery used `kv_pools`, which is intentionally empty in unified
V4. The backend incorrectly declared C4/C128 absent and dereferenced a None
C4 table. Corrected discovery to use the logical source map only for unified
storage, retaining actual paged allocations for V4.1. Four CPU constructor
regressions plus existing V4.1 suite:132passed and16subtests. No GPU request
has run yet; third fresh startup is next.

Third process506227 captured all target tiers1/8/32, then failed at the same
expression for a different legitimate layout: the bundled DSpark draft is
SWA-only. Added absent-compressed-stream handling (SWA still updated), and
single-ratio/empty-ratio CPU tests. Combined ratio tests, graph-shape audit
tests and V4.1 CPU regressions:144passed plus16subtests. User then narrowed
the task to AR-only, so no further DSpark startup/benchmark is planned.

Next service uses accepted native M32 overlap/gate-prefetch/down-uniform/fixed
warmup/AR4 plus the existing large-prefill throughput profile, pool1M and
explicit graph tiers1/2/4/8/16/32/64. Quant capacity override0 selects the
general grid instead of underallocating C64*Top6 rows. A host-only deduplicated
graph-key audit records actual executed rows separately from request count.

AR process519441 launched on127.0.0.1:30021 in tmux session
`dsv4-tp8-ar-20260913`. Controller `run-ar-pilot.py` waits on this exact PID
and birth time, checks GPU ownership, then runs France and natural-output
C1/C8/C32 warmups plus two short measured waves each. Full seven-tier,
three-round matrix starts only after inspecting that pilot. No new speed yet.

## Native fresh startup passed

At23:42HKT AR process519441 reached ready. Seven target-decode tiers captured,
no target-verify/draft graphs. Requested1M pool retained; before graph capture
about24.6GiB/GCD remained. France returned `The capital of France is **Paris**.`
with completion IDs and no speculative acceptance. Runtime audit confirms
C1 uses key1/executed_rows1/input_rows1, not a padded C32 graph. Initial
request includes cold module/JIT work and is not a speed result.
The existing CK scale-layout CPU oracle also passed all40 mutations across
TP4/TP8 gate/down layouts. Native short pilot and full P/D remain pending.
