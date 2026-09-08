# TP8 subgroup8 down row-prefetch oracle

Base d42ed1ae10. No production selector changes. Existing TP4 down prefetch
has an explicit subgroup16 static_assert; K256 needs subgroup8. Copied into
independent gfx90a_fp4_tp8_down_prefetch_oracle.cuh and changed only that guard
and its comment, not the original TP4 header. Reduction bounds, assignment
mapping and scale addressing already derive from K/32. No logical-scale cache,
weight repack, or new persistent allocation. Tested existing shuffled scales.

GPU4 only, amd-smi no external PIDs; retained service2821603 idle. Synthetic
transient gate/down weights ~384MiB, freed on exit. No HTTP benchmark overlap.
Standalone script --down-prefetch holds accepted gate-prefetch fixed on both
arms and changes only down. Post-sort chain includes gate/intermediate quant/
down/fixed-slot reduction; excludes initial quant/sorter. Five ABBA cycles,
100replays per sample,20warmups. Original TP8 A4/R2/W8/G832/D832 geometry.

Second run explicitly checks FP32 partials (first run only checked final BF16):

|Active experts / scans|A chain us|B chain us|Saved us|
|---|---:|---:|---:|
|133/133|274.9263|265.7904|9.1359|
|104/105|241.2568|235.1578|6.0990|
|32/61|182.7717|179.7531|3.0186|

All three distributions: gate BF16, down FP32 partial and final BF16 exact
100/100 under mutation; max final error0; candidate final stable across ten
replays per mutation (1000total). No full-model semantic claim from synthetic
weights. Raw samples and witnesses in adjacent JSON. First independent timing
run also saved~3.2–8.9us. Not an E2E result; do not add component gains to rates.

Next: default-off TP8 native M32 down selector, retaining the original scale
layout and fixed reduction; service ABBA with gate-prefetch ON in both arms.
Production remains unchanged until scope tests and real C32 correctness pass.

## Default-off service experiment pending

Added SGLANG_DSV4_GFX90A_TP8_M32_DOWN_PREFETCH. Shares native M32 eligibility
with gate but independent activation: down-only does not turn on gate or legacy
AR. Runner checks the previously tested TP8/gfx90a/M32/I256/A4/R2/G832 shape,
plus down832/R2 and no logical-scale or down-consumer override. Public wrapper
adds only M32/K256/W8 to its prefetch cases and selects the isolated header.
No original TP4 header change, scale cache, weight cache or new workspace.

Ten CPU tests pass (scope, gate/down selectors, independent flags, cleanup).
Running exec22627; prefix /tmp/dsv4_tp8_down_prefetch_abba_20260908,
arms DPA1:0 DPB1:1 DPB2:1 DPA2:0. Helper --gate-prefetch-on explicitly holds
accepted gate ON in both arms (parent control had gateOFF). --france-c32 each
arm, C1 benchmark, six diverse32code waves,12transition checks. Existing C4
overlap and legacy AR remainON, pool131072/mem0.80/interleave-all unchanged.
amd-smi confirmed no external GPU PIDs. Final control stays resident. Verify
live state/PID before waiting; do not interpret a readiness observation timeout
as a dead process. No E2E promotion before complete ABBA and correctness.

## Withdrawn at user request

User requested withdrawal on 2026-09-08. Stopped ABBA controller 2836535;
service 2844486 has exited and no sglang.launch_server process remained on
inspection. No replacement service was started. Reverted only this experiment's
uncommitted production selector, wrapper, helper and test changes to HEAD.
Accepted gate-prefetch, legacy AR and C4 overlap code remains untouched, as do
unrelated worktree changes. Standalone oracle and historical data are retained.
DPA1 recorded 984.4644 E2E tok/s; DPB1 recorded 990.0993, but ABBA was not
completed: these are partial observations, not an accepted performance gain.
The earlier pending/final-control statements above are superseded by this note.
