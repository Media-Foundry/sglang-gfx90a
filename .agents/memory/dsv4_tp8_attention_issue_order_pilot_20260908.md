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

## Return control completed: no benefit, retain order0

PID3611667 completed the same validation and six-wave benchmark. This gives
fresh-process **ABA screening**, not a full ABBA or statistical proof:

| Order | PID | C1 trimmed/geomean | HTTP C32 warm | Resident C32 warm |
|---|---:|---:|---:|---:|
|0 before|3589516|83.413359|986.486947|1034.427640|
|3 pilot|3603541|83.780409|985.428574|1032.847084|
|0 return|3611667|83.656105|985.152692|1031.135067|

Against the geometric mean of the bracketing order0 processes, order3 is
approximately +0.0065% resident and -0.040% HTTP: effectively neutral, with
visible process drift. C1 was not the target of this M32-only stream ordering;
do not credit its slight variation as an optimization gain.

Decision: stop order3 for the current TP8 workload and retain order0. The
TP4 historical benefit does not reproduce here. No production model/kernel
changes or defaults were introduced. Do not spend another multi-service sweep
on orders1/2 without new critical-path evidence; TP4 already rejected them.

All three processes: France32/32, 12 measured C1 full outputs matching the
existing reference (36 total), recomputed C1 uint32 hashes. All576 C32 outputs
have256 tokens, finish=length, valid recomputed JSON completion hashes and no
speculative acceptance statistic. Across-six-wave exact counts6/32,5/32,9/32:
still not general semantic or bitwise-determinism proof.

Return process reports1M pool/context, graph memory0.65GB/GCD; AMD audit after
completion found only the18 distinct owned GPU PIDs. No controller remains.
The current live service is PID3611667, order0, down-uniform1, fixed warmup1,
paired graphs0, loopback30011. Model/KV configuration unchanged.

Return artifact stem:
`/tmp/dsv4_tp8_issue_order0_return_validation_20260908.block0`
- `.c1.json`: `91964a4e4de0b90ceb5127df660c1429650f4d838b6e62eccd3e5d6e2a7086e6`
- `.c32.json`: `21b21582944fd3cb9aeb87f8ca8f1fe1b4560d60e7d9b53405a26537084f9280`
- `.france.json`: `7f29267cc75b84322f7bc00cc8612c5ab996bcbeb8cd8aebd7468e8d549106fc`

Pilot artifact SHA256s under the previously listed stem:
- `.c1.json`: `49c1f8a4c3fd67c88518ffe3e667c6c4070ffcf2e6ef9ac413bf5128f6e9f7c8`
- `.c32.json`: `ff45611a55f7f7b42ffd103bdcf6c63db502cd30dc69d8d1ee8d3c6e9bc196a5`
- `.france.json`: `16c225019b5441271fb70b448bde0292bee0f92177fe8f69f97e29655dac22cb`
