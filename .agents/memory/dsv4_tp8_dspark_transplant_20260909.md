# TP8 DSpark transplant — in progress

**B1 anchor-only transplant REJECTED:** natural-code warmup collapsed into
repetition despite a correct France answer. Do not use the opt-in approximate
profile as a recommended TP8 setting. See rejection details below.

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

## Follow-up while B1 runs

B1 reached ready, France returned Paris, and C32 warmup subprocess 4054542
was verified live under service 4046629. Session 84644 remains the owned
gate/benchmark handle; do not restart it based on an observation timeout.

Prepared a separate default-off TP8 M128 target attention overlap selector
(`SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_ATTN_MULTISTREAM`). Stream allocation
and forward selection both recognize it. Its pure guard requires HIP, TP8,
C4, M128, C32, TARGET_VERIFY, width four and unified KV. Unit negatives
cover AR/non-target, TP4, C33/M132 safety graph, wrong width/backend, and
disabled state. One parameterized-loop test passes; Python AST parses.
This code is not enabled in B1 and has not passed GPU/E2E validation yet.
Do not stage unrelated pre-existing edits in deepseek_v4.py when committing.

CK audit additionally confirms the C++ wrapper hardcodes H16, and the core
uses M16 MFMA with Q accesses for all 16 heads. Merely accepting H8 at the
wrapper would read out of bounds. A masked/padded head implementation needs
its own measured oracle; no CK selector has been broadened.

## B1 result and B2

B1 warmup completed at 562.9517 output tok/s over 113.6758 resident seconds
(63,994 resident tokens). The first four reviewed requests show repeated
`arg_utils` imports, repeated scheduling phrases, fabricated repeated import
lists, and repeated allocator paragraphs. All reached the 2048-token cap;
acceptance around 3.82–3.92 is NOT correctness evidence. This is material
semantic collapse, not tolerable low-bit drift. Formal B1 rounds were stopped
after discovering the warmup failure; do not report a three-round B1 median.

Stopped exact owned controller 4048927 and service 4046629. B2 service PID
4057043 enables only the new TP8 target attention overlap while disabling
anchor-only and pre-router compaction, restoring graph max32 and all full
routed target rows. Output root `/tmp/dsv4_tp8_dspark_overlap_b2_20260909`.
Service log has the same prefix plus `.service.log`. The generalized gate
script is now called with PID and output root as arguments. B2 is unmeasured
at this update. Keep the full migration goal active.

Quantified repetition diagnostic (not a general correctness oracle): in the
last 512 output IDs, compute the duplicate fraction among sliding 8-grams.
Full-target C32 warmup median 0.065, 0/32 above 0.75, 10/32 hit length cap.
B1 median 0.949, 29/32 above 0.75, all 32 hit the cap. This supports the
manual finding of widespread collapse, not just a different greedy hash.
