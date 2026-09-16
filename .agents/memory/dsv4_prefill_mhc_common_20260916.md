# Common large-prefill MHC: runtime connected, component passed, live check running

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
forwarding. New helper/kernel wiring still uncommitted at this note.

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
identical prefixes. No assumptions of legacy equality. run.py will now start
after this passing diagnostic is checkpointed. Scope extends only
large prefill; follow-up tail/decode regression still required.

Runtime source must now stay frozen during service tests. User/runtime pickle
and unrelated untracked files preserved. Persistent goal active.
