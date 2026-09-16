# 32K wide owner: +38.0224% over legacy MHC, numerical checks passed

Base commit28a25753c0. Separate workload from accepted8K10205.900816 and
16K10024.774291 results. This trial uses frozen public-code C16x32K prompts,
524286 actual input tokens, zero prefix hits, native AR/TP8/EP1/original
checkpoint,1M logical KV and32768 chunk. No production code changes this turn.

Directory `.agents/experiments/dsv4_wide_owner_32k_20260916/`.
Driver adapted from prior16K trial, same explicit launcher flags and numerical
checks. Both arms enable full wide-query16/runtime-M fallback, producer off;
only QUERY_OWNER_WIDE differs. Three scored waves per leg, four128-token quality
waves per process, fixed64-token A1 continuation, three processes A1/B/A2.

## Critical scope finding

Same launcher flags do NOT mean all16K computational paths carry over. With
one32K request per forward, global_batch_size=1 admits the older MHC fused tail
before `gfx90a_mhc_pre_mix_from_partials_triton` reaches paired/owner dispatch.
The default FP16_MHC_DOT=1 selects a runtime FP16 copy of Fn here. All8ranks in
the diagnostic explicitly logged path=fused_tail,rows32768,batch1,FP16 weight.
Timing-paths.json records this before teacher-forced mixed chunks can change
the dispatch; premix_owner_ranks=[] in diagnostic. Fixed CK, H16, direct exact
dequant do hit; their scope is the overall large-prefill context, not the actual
pre-mix function selection.

Checkpoint files/quantization unchanged, but do NOT label32K as the same FP32
MHC arithmetic used by the accepted multi-request16K path. This is preexisting
behavior common to A and B, not caused by wide-owner. It is a concrete source
of batch-dependent precision, not proof of the sole cause of historical drift.

Further source audit: fused-tail uses MHC_SINKHORN_ITERS directly, whereas
hc_split_sinkhorn's ordinary non-batch1 native branch takes20. Actual live B
process2531860 environment read via psutil confirms ITers=8,FP16_MHC_DOT=1,
PREFILL_MHC_CONFIG_ITERS=0,NATIVE_MHC_POST_PRE_FULL=0,NATIVE_MHC_POST_PRE=0.
Thus the legacy32K baseline also uses8 versus20 Sinkhorn iterations; the
difference is NOT only floating-point ordering/precision. Do not promote its
eventual speed as a strict20-iteration reference checkpoint. Finish the frozen
indexer comparison, then unify large-prefill dispatch as a separate experiment.

Next after isolated indexer trial: investigate a narrowly scoped ordinary
large-prefill policy that avoids legacy batch1 priority, preserving FP32 Fn
and using the already measured paired/owner chain. It must be validated
against proper FP32/teacher references, not claimed bit-exact to FP16 legacy.
Native small-C1 decode and DSpark must stay untouched. Do not change these
runtime files while the current ABBA is live.

## Real-model diagnostic completed

check/complete.json:2688=21 C4layers x16forwards x8ranks exact comparisons of
full/owned score bytes and logical/physical Top512; all pass. France passes,
1M pool/32Kchunk confirmed. Diagnostic rate5659.320 includes duplicate scoring
and synchronizations; not a service score or regression. Diagnostic service
2511876 stopped cleanly. Formal run.py has now been started; inspect run.log,
per-arm driver logs and actual process state. No formal result yet at note
creation; do not infer completion from this note or existing scripts.

## Completed ABBA: indexer-only delta over legacy MHC

| Leg | warm median input tok/s |
|---|---:|
|A1|6167.099820|
|B1|8506.753721|
|B2|8508.078102|
|A2|6160.489164|

Three scored waves per leg; control6163.794492,candidate8507.415911,
**+38.022381%**, control drift-0.107192%. Source/input hashes frozen and raw
client interval formula independently rechecked; zero prefix hits and full
prompt echoes. Checkpoint files unchanged,1M pool retained. Label this speed
**legacy FP16-Fn/8-iteration MHC**, not the accepted16K FP32/20 arithmetic.

All12 quality waves (three processes x4 waves x16 requests=192 answers) match
all128 output tokens within/across configurations. Both A1/A2 and A1/B fixed
continuations compare1008 non-null positions: max/mean logprob delta0,
1008 Top1 matches and1008 complete Top5 records identical. No empty outputs
or leading nulls counted. All16 unique excerpts read: coherent, no obvious
garbling/loop collapse, but contain factual errors (e.g. expansion of MHC and
hardware-family classification). This is numerical non-regression evidence,
not a factual/code-correctness certification.128 tokens are truncated starts.

All four services stopped cleanly; final amd-smi confirms no GPU processes.
Evidence summary.json/acceptance.json/service-evidence.tar.gz plus per-file
manifest. measured-legacy-launcher.sh deliberately names the approximation;
no global default promotion or claim of FP32/20 acceptance.

Read-only follow-up audit is in `mhc-next-audit.md` in the experiment directory.
It covers all singleton-priority branches, not only fused-tail, and explains
why disabling just FP16 is insufficient (Sinkhorn8 remains). No runtime policy
change was made during this comparison. Session34055 exited0, analyzer complete.

Next common-MHC prototype is outside runtime at
`.agents/experiments/dsv4_prefill_mhc_common_20260916/policy.py`. Four CPU-only
policy tests pass, but it is not connected or GPU-validated yet. Existing
strict scope/no-singleton hint avoids mutating ForwardBatch and rejects
conflicting BF16/MFMA/iteration overrides. Integration can begin only now that
the frozen service trial is finished; keep numerical-reference labels explicit.

Runtime pickle and unrelated files preserved. Goal active.
