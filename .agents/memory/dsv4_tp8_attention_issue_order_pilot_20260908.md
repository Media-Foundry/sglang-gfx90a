# TP8 C32 compressor issue-order pilot

2026-09-08, production source unchanged. Existing HIP multistream helper
reads `SGLANG_DSV4_GFX90A_TP4_M32_ATTN_ISSUE_ORDER` despite its TP4 name;
the accepted TP8 C4/M32 native multistream branch calls the same helper.
Live prior process PID3589516 had this variable unset (default0), while TP8
M32 multistream was1. No TP8-specific issue-order experiment was found in
the inspected TP8 records. TP4's historical order3 win is not TP8 evidence.

Mode0 submits core/indexer compressors before q-lora projection; mode3
submits q-lora first, then those same compressors with their stream events.
No arithmetic, weights, cache layout, graph tier or collective changed.
Native TP8/EP1/no-A2A; down-uniform1 and fixed-first-use warmup1 held constant.
1M KV unchanged. AMD PID ownership audited before replacing the owned service.

## Mode3 first process (not ABBA)

PID3603541; four C1 rounds, three fixed diverse code prompts; C32 six waves
of the existing 32-request diverse-code workload, 256 output tokens each.

- C1 per-task trimmed/geomean: 83.78040870687951 tok/s.
- C32 warm HTTP median: 985.4285740181829 tok/s.
- C32 warm resident median: 1032.8470836038346 tok/s.
- France32/32; 12 C1 full sequences match the established reference.
- All192 C32 length256/finish-length/hash-integrity checks pass, no speculative
  acceptance statistics. Only5/32 cross-six-round exact: not deterministic.

Adjacent prior order0 A2 was C1 83.41336, HTTP986.48695, resident1034.42764.
The candidate does not show an initial C32 improvement, but independent
process variability prevents a strong negative attribution from this alone.
Restore order0 and collect return-control measurements. Do not promote order3.

Raw state:
`/tmp/dsv4_tp8_issue_order3_pilot_validation_20260908.json`
Artifacts append `.block0.{france,c1,c32}.json` to the stem.
Service log: `/tmp/dsv4_tp8_issue_order3_pilot_20260908.service.log`.
Private environment snapshot is excluded from version control.

Harness now supports explicit issue-order overrides only in the guarded
native TP8/1M fixed-warmup launch mode. Validation records the actual issue
order; down-ABBA aggregation rejects mixed issue orders. Six CPU auditor
tests (including mixed-order rejection) and Python syntax checks pass.
No production defaults changed. Return-control launch is separately recorded
under `/tmp/dsv4_tp8_issue_order0_return_20260908.json`.
