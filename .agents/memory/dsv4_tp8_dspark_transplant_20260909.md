# TP8 DSpark transplant — in progress

## Baseline

TP8/EP1/no-A2A, original checkpoint, gamma three, 1M logical token pool,
full routed target verification. Real open-code natural-EOS workload,
three rounds with at least 30 seconds of common resident decode per round.
C1/2/4/8/16/32 medians: 97.4304 / 161.6553 / 271.6058 / 524.4104 /
729.8458 / 879.4233 output tok/s. France answered Paris.
Artifacts: `/tmp/dsv4_open_code_pd_tp8_dspark_20260909/`.

## First transplant

Default-off `SGLANG_DSV4_GFX90A_DSPARK_TP8_BS32_PROFILE=1` in the launch
script enables existing TP-agnostic M128 anchor-only routed selection and
pre-router compaction. It adds BS33 for the position-20 early exact guard.
This is approximate target computation after the guard, NOT a lossless
speculative or native-AR claim. No checkpoint weights are modified.

Profile rejects AR and incompatible parallel layouts; disables native-only
fixed down warmup and paired graphs. `bash -n` passed and three existing
anchor-only guard tests passed. The AR-command negative test exits 1 with
an explicit profile error. These tests do not establish TP8 numerical quality.

Candidate B1 PID 4046629; startup log:
`/tmp/dsv4_tp8_dspark_transplant_b1_20260909.service.log`.
Gate/benchmark script: `/tmp/dsv4_tp8_transplant_b1_gate.py`, exec session 84644.
Results will be under `/tmp/dsv4_tp8_dspark_transplant_b1_20260909/`.
It waits for readiness, checks France, then runs warmup plus three C32 rounds
using the identical code manifest. Inspect real output text and acceptance
before accepting any speed gain. No candidate performance available yet.

## Remaining transplant work

- TP4 CK sparse attention wrapper hardcodes H16 in both validation and args;
  TP8 has H8. Do not widen the Python selector without an H8 component oracle.
- TP4 target attention overlap has explicit attn_tp_size==4 guards at stream
  allocation and execution. Needs a separate TP8 opt-in and graph/correctness
  testing; not enabled in B1.
- TP4 AR CTA tuning is not automatically optimal for eight peers. Measure
  complete graph/E2E before adopting its 1-MiB CTA count.
- Final acceptance needs matched-workload multi-round E2E and AR negative
  controls, and must separately label approximate vs full-target results.
