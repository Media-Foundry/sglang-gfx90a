# Down-uniform: same-process ABBA confirms small C32 gain, C1 neutral

Controller32030 completed on one PID3527138; no process restart between arms.
Original checkpoint, native TP8/EP1/no-A2A,1M KV, same fixed heterogeneous code
corpus. Candidate/control/control/candidate; final arm restored to control and
confirmed in all8rank logs. Service remains live. Summary auditor passed.

| Metric | Control | Candidate | Change |
|---|---:|---:|---:|
|C1 per-case trimmed/geomean tok/s|83.443274|83.462278|+0.0228%|
|C32 warm HTTP tok/s|985.402509|989.933320|+0.4598%|
|C32 warm resident tok/s|1031.046735|1036.516726|+0.5305%|

C1: four measured rounds per each of three real tasks; drop min/max per task,
then geometric mean across tasks and two blocks per arm. C32: six waves/block,
drop wave0, median of five remaining waves, geometric mean of two blocks/arm.
These are ABBA point estimates, not confidence intervals.

## Correctness and capacity

- Four separate FranceC32 tests:32/32 answer-through-EOS prefixes each.
- All48 measured C1 full completion ID lists match the fixed historical C1
  reference; summary independently checks cross-arm IDs and recomputes hashes.
- All768 code requests:256tokens,finish=length,hash recomputed from output IDs.
- C32 cross-round full-output exact counts by block:8/32,8/32,7/32,5/32.
  Neither arm is fully bitwise deterministic. Completion checks and France do
  not establish general semantic correctness or numerical equivalence.
- Runtime1,048,576-token pool retained. Paired diagnostic capture had36MiB
  immediate extra device allocation/GCD; total logged graph0.68GB versus the
  normal baseline0.65GB. Arm switches passed the post-registration device-free
  budget check. This is diagnostic memory, not a proposed production cache.

## Interpretation / next acceptance step

Previous restart-based ABBA runs also showed about0.55--0.64% resident benefit,
but their C1 slowdown was confounded by process state. Holding the process and
C1 graph fixed removes that observed slowdown: C1 here is effectively neutral.
The two arms still have distinct M32 graph allocations, so do not pretend this
eliminates every allocator-related confound. Three runs consistently support
a small C32 improvement, not a major throughput breakthrough.

Next restore a **single-graph** production-shaped service: paired flagOFF,
DOWN_UNIFORM=1, same exact launch snapshot and1M KV. Validate France, C1 fixed
IDs, C32 real-code throughput/correctness and graph allocation before deciding
to add the flag to an accepted TP8 profile. Do not enable paired graphs by
default or advertise the extra graph itself as a speed optimization.

## Artifacts

State `/tmp/dsv4_tp8_paired_same_process_wire_abba_20260908.json`;
summary `/tmp/dsv4_tp8_paired_same_process_wire_summary_20260908.json`.
Block `.c1/.c32/.france.json` and logs use the state stem plus `.block0..3`.
Adjacent committed JSON contains per-block metrics, workload fingerprint and
SHA256 for every result artifact. Raw token IDs/text remain in those artifacts.
