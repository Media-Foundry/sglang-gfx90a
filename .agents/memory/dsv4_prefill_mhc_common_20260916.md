# Common large-prefill MHC: 32K service ABBA accepted, 9605.07 input tok/s

## Final authoritative result

All arms completed and owned services stopped. Formal non-diagnostic ABBA,
three scored waves per leg, C16x32K / 524286 input tokens / zero prefix hits:

| Leg | Median input tok/s |
|---|---:|
| A1 full paired FP32/20 | 8665.983485 |
| B1 TP pre-mix owner FP32/20 | 9607.475082 |
| B2 TP pre-mix owner FP32/20 | 9602.655708 |
| A2 full paired FP32/20 | 8668.883595 |

Control center **8667.433540**, candidate **9605.065395**, **+10.817872%**.
Control drift +0.033465%. This gain is owner compute distribution against the
same FP32/20 contract, NOT a numerical comparison against legacy FP16/8.
All192 128-token continuations identical per request across three processes
and12 waves. Both newA1/A2 and newA1/B have1008/1008 identical selected-token
logprobs, Top1 and complete Top5 records on the same fixed continuation.
Legacy/B retains the separately disclosed nonzero differences below.

Acceptance gates source hashes, owned cleanup, all-rank path selection,
10880 live full-reference comparisons, raw timing reconstruction and nonempty
numerical evidence. See acceptance.json, summary.json, archive-manifest.json,
service-evidence.tar.gz and validated-launcher.sh in this experiment directory.
The launcher is a byte-for-byte copy of the measured candidate launcher;
it does not rely on an outer export surviving older launcher overrides.
No global defaults changed. Small prefill/decode unchanged and universal batch
invariance unproven. Follow-up8K/16K common-policy regression and latest profile
are still required. Historical8K10205.90 and16K10024.77 are different length
workloads, not evidence that32K9605 is a same-workload slowdown.

The following sections preserve the component evidence and chronological notes;
their in-progress statements are superseded by this final result.

Base HEAD7310f82ed1 (pushed) records32K indexer+38.0224% over legacy FP16/8
MHC. Do NOT treat8507.415911 as a FP32/20 reference. Prior accepted8K10205.9
and16K10024.8 used the actual multi-request20-iteration path.

New default-off flag `SGLANG_DSV4_PREFILL_MHC_COMMON_FP32=1`:
`gfx90a_mhc_prefill_policy.py` resolves only this MHC call's singleton dispatch
hint toNone inside the existing strict mix-pair scope/M8192..65536. It does
not mutate ForwardBatch or fake a different scheduler batch. mhc.py applies
it at entry to the gfx90a/H4096/HC4/sinkhorn_repeat20 branch, BEFORE all single-
request shortcuts. Thus full-native, fused-tail, splitK, native-finish and fused
weighted-RMS priorities all bypass consistently; FP32 paired/owner +20-iteration
Sinkhorn + existing weighted sum/RMS remain. BF16-dot,MFMA pre-mix,custom
iterations andcomb-refinement overrides raise rather than silently mix.

Scope excludes other models (includingV4.1),TP4,CP/TBO,draft/speculative,decode,
graph capture andM<8192. **Small prefill tails and native decode still have
legacy arithmetic; this is NOT a global whole-model batch-invariance fix.**
Goal remains open. Consider extending the correctness policy with separate
tail/decode evidence later; do not infer that large-prefill success proves it.

## Component evidence

Directory `.agents/experiments/dsv4_prefill_mhc_common_20260916/`.
oracle-v2.json complete,PCI0000:b3:00.0,HIP_VISIBLE_DEVICES=5 (rocm-smi card7).
Full mhc_fused_post_pre tested with repeated real activation seeds and actual
checkpoint Fn/base/scale/norm fromlayers0/20/42. These seed tensors are not
asserted to form correct causal inputs for each layer; this is boundary math
validation, not model-quality evidence. No checkpoint tensor/files modified.

Shapes1,17,8191,8192,8193,32767,32768,65536 atlayer0;8192/32767 atlayers20/42.
Three mutations + reversed row ordering per shape. All four outputs byte-exact
between newbatch1 and oldbatch2 FP32/20 reference for admitted shapes; small-M
candidate unchanged from oldbatch1. All finite. Single-GCD test disables TP
premix owner and replaces only distributed allocation wrappers with nullcontext.
No arithmetic kernels replaced. This is eager-prefill evidence, not graph test.

M32767 full boundary: legacy10.525..10.594ms,common9.308..9.440ms (without
cross-rank owner). Layer0 M32768:10.5847 ->9.4004ms. Not E2E numbers.
Legacy/candidate differ as expected: residual0; post max2.49e-5..1.22e-4;
comb up to0.15254; normalized layer input up to0.015625. These differences
include8vs20 Sinkhorn, NOT merely FP16 rounding. Do not demand equality to
legacy or pretend arithmetic semantics were unchanged.

First oracle attempt lacked AOT PYTHONPATH and exited1 with missing sgl_kernel
when entering reference Sinkhorn; oracle.log and partial oracle.json are NOT
passing results. v2 adds explicit AOT paths, output argument and failed-status
recording; no failed evidence overwritten. Full tested source hashes inv2json.

CPU tests:28 passed,2 skipped,13 subtests; one preexisting asyncio_mode config
warning and14 torch.jit deprecation warnings. unit.log stores exact output.
Includes strict scope, hint contract, conflicts and existing AMD MHC metadata
forwarding. Helper/kernel wiring and checks committed/pushed as356f4dc0c0.

## Real-service validation in progress

service.py --arm check completed, owned process2551662 stopped, session89794
exited0. Candidate:commonFP32=1,premixowner=1,ownercheck=1,wideindexerowner=1,
all latest exact post/H16/uniqueSet/vec4/dequant settings,1M pool/C16x32K.
**10880=85 fused boundaries x16 forwards x8rank** full-reference mix checks
passed. All8common/owner hits; NO large-prefill legacy splitK. France passes,
1M pool and32K chunk verified. Diagnostic8028.3399 includes full duplicate
reference and synchronizations, NOT a scoring result. Owned service stopped.

Planned formal service.py arms A1/B/A2 usecommonFP32=1 in BOTH arms and only
switch premix owner0/1. This provides the proper FP32/20 numerical reference.
Fixed continuation is deliberately frozen from legacy32K A1: analyzer can
compare A1/A2,B exact and also separately quantify legacy/B divergence on
identical prefixes. No assumptions of legacy equality. run.py is now running,
session30486, parent2559164; A1 service2559175 confirmed live. A1 warmup8272.05;
three scored waves8669.791041/8665.983485/8665.934539, median8665.983485.
This is full paired FP32/20 without TP pre-mix owner. Formal B/A2 not completed
at this update; do not infer final gain. A1 log confirms all8 ranks selecting
common path for scheduler_batch1,dispatch_hintNone. Scope extends only
large prefill; follow-up tail/decode regression still required.

Further progress: A1 finished all four quality waves with16/16 repeat equality,
teacher forcing saved and service2559175 stopped. Its16 free continuations all
differ from the legacy32K baseline (common prefix lengths21,3,8,0,2,6,23,35,0,
2,0,0,0,20,0,0). All16 excerpts read: coherent code-review starts, no obvious
loop/garble; generated bug claims not certified. This is NOT a tiny-rounding
change or an answer-quality score. On identical legacy-frozen continuations,
newA1 versus legacyA1 at1008 positions: max selected-token logprob delta1.239357,
mean0.04663086, Top1 matches968/1008, complete Top5 records match441/1008.
This is expected semantic/precision restoration (FP32/20 versusFP16/8), not
evidence of a failed exact optimization. B MUST match newA1, not legacy.

B process2567203 confirmed live; first unscored warmup9107.8733. B1 three scored
waves9607.953840/9604.225678/9607.475082, median9607.475082 (~10.9% over newA1).
B2/A2 and cross-process numerical gates still pending at this update.
Orchestrator remains session30486, no restart needed.

Runtime source must now stay frozen during service tests. User/runtime pickle
and unrelated untracked files preserved. Persistent goal active.
