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

## H8 oracle preparation (unvalidated)

Prepared `scripts/rocm/bench_dsv4_sparse_h8_oracle.py` and a separate FFI
wrapper `gfx90a_dsv4_sparse_h8_oracle.cuh`. The existing MFMA split core now
has a default-16 template head count, with H8 specializing masked Q loads and
partial writes. No production Python attention selector has been widened.
The original H16 wrapper continues to instantiate the default-16 core.

The planned oracle checks lengths 0/17/128/512, ragged rows, ten Q/sink/index
mutations against FP32 torch attention, graph/eager mutation equality and
1000 identical graph replays, plus GPU-event timing. Python syntax passes;
HIP compilation and GPU validation have NOT run. Do not enable it in service
until those pass and complete stage timing beats the current H8 Triton path.
Wait for B2 E2E to finish before using any GPU for the oracle. B2 controller
session 46284 and warmup process 4064463 were verified live.

B2 warmup subsequently completed at 877.9341 tok/s (12.7333 resident seconds).
Repetition diagnostic: median 0.0733, 0/32 above 0.75; reviewed outputs no
longer show B1's looping collapse. Formal benchmark PID 4066124 is live;
first wave 886.9 tok/s over 9.7046 seconds, not a complete 30-second round.

Session 79768 runs `/tmp/dsv4_after_overlap_h8_oracle.py`: waits for that exact
benchmark PID/birth and a complete result, rechecks GPU PID ownership, then
stops only B2 service 4057043, runs the isolated H8 oracle on physical GCD4,
and restores the full-target overlap service in a finally block. This avoids
GPU test interference. Inspect the session output for oracle failures and the
new restored PID. Restore log:
`/tmp/dsv4_tp8_dspark_overlap_restored_20260909.service.log`.
Oracle output (only written after all checks):
`/tmp/dsv4_sparse_h8_oracle_20260909.json`.

Further port audit: the installed AIter `custom_all_reduce.cuh` guards both
`AITER_GFX90A_AR_512K_BLOCKS` and `AITER_GFX90A_AR_1M_BLOCKS` by
`world_size_ == 4`; setting those on TP8 has no effect. Likewise TP4 gate-row
prefetch is nested under TP4-specific shape predicates; the TP8 M32 prefetch
alternative requires native_m32_active and does not automatically cover the
DSpark draft. These need bounded TP8 oracles, not copied env exports.

The H8 oracle now compiles H16 and H8 exports in the same new JIT module and
requires H8 output to exactly match the first eight heads of an H16 input
with matching Q/sink. It also compares timing/error against the current H8
Triton entry. This remains queued, not a completed validation.

## B2 complete

Three round rates: 877.951849 / 876.887980 / 880.195032 output tok/s;
resident windows 35.5953 / 40.8228 / 35.4603 seconds. Median 877.951849,
versus full-target baseline 879.423277 (about -0.17%). No demonstrated speed
gain; keep overlap opt-in, not recommended by default. Across 288 responses,
tail 8-gram duplicate fraction median 0.0584, none above 0.75. France passed.
This is a looping-collapse screen, not a semantic correctness proof.

Initial H8 oracle compiled but rejected its length-zero case: a zero-element
index tensor supplied a null pointer to the existing ABI validator. Fixed the
fixture to allocate a one-element unused sentinel while leaving indptr zero.
Added hipGetErrorString diagnostics. No numeric failure was established.
Stopped the auto-restored service 4081881 after ownership checks, rerunning
the oracle in session 77815 on GCD4 only. Finally restores the full-target
overlap service to `...overlap_restored2_20260909.service.log`; inspect session
output for the new PID and actual oracle result.

## H8 oracle passed; E2E integration still pending

Retry session 77815 exited zero. All fixtures passed FP32 torch tolerance
atol=0.004, rtol=0.02 (not absolute error <=0.004), H8 vs H16-first-eight
bitwise comparison, ten graph/eager mutations and 1000 graph replays.

| max row keys | H8 CK median us | Triton median us | max abs vs FP32 |
|---:|---:|---:|---:|
|0|26.616|13.705|0|
|17|36.915|32.735|0.0081234|
|128|57.609|103.995|0.00350094|
|512|137.297|386.442|0.00190940|

M128/H8/D512, one quarter of rows have half the specified key length.
Five GPU-event timing batches each, CK then Triton (not ABBA); treat these
as a promising component screen, not a final speedup checkpoint. Short rows
regress; no production selector changed yet. Original H16 math is retained.
Raw JSON: `/tmp/dsv4_sparse_h8_oracle_20260909.json`.
Full-target overlap service restored as PID 4086117, log
`/tmp/dsv4_tp8_dspark_overlap_restored2_20260909.service.log`.
Next: guarded TP8 target-only H8 integration, actual-path-hit validation,
real-code E2E and AR negative control, with no anchor-only approximation.

## C3 service integration (in progress)

Independent default-off `SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_SPARSE_DECODE`
is consumed in the HIP backend where the actual ForwardBatch is available.
First service candidate covers C128 only, TP8, C32/M128, target verify width4,
gfx90a, BF16 H8 and no fused inverse RoPE. Other shapes/modes fall through to
the existing implementation; no AR-wide shape selector is changed. The H8
FFI uses the exact oracle module. Metadata/indices and KV store are unchanged.
Two CPU guard tests pass, including non-target and BS33 negatives; AST checks
pass. A one-time log records actual H8 path hits.

Stopped idle restored B2 service 4086117 after AMD PID ownership checks.
C3 PID 4093679, log `/tmp/dsv4_tp8_dspark_ck_c3_20260909.service.log`.
Gate script starts France then C32 warmup plus three formal rounds, output
directory `/tmp/dsv4_tp8_dspark_ck_c3_20260909`. No C3 result yet. C4 attention
integration and all-reduce/draft-prefetch transplantation remain unverified.

C3 startup and France passed, but there were no CK hit logs. Source audit
found `_local_attn_sink()` returns a legacy padded 64-element vector with
the real local eight heads first. H8 wrapper's exact (8,) sink check silently
fell back. Stopped that benchmark; it is NOT a CK E2E result. Wrapper now
slices the first eight sink entries without allocation or arithmetic changes.
C3b restart uses `/tmp/dsv4_tp8_dspark_ck_c3b_20260909.service.log`; actual
path hits must be checked before accepting its timing.
