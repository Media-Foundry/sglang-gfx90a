# Small prefill MHC contract: component evidence and opt-in intervention

Motivation:32K route-producer ABBA found historical teacher drift. Fresh route
candidate/control match all1008 positions,so route is not the cause in that
pair. Historical/fresh chunk layouts differ. Existing common policy explicitly
excludes M<8192; singleton tails use FP16 mixing/8 Sinkhorn iterations while
large common path uses original FP32 Fn/20 iterations. This is more than ULP
noise. Does not yet prove MHC explains every historical whole-model difference.

Independent oracle-v1(session80474 exit0) uses real activation seeds repeated
to several M and real checkpoint Fn/base/scale/norm at layers0/20/42. They are
not claimed as causal inputs for each layer.12 cases:legacy/common residual
exact,but comb max difference up to0.15317,normalized BF16 up to0.0078125.
Explicit common small rows match corresponding repeated M8192 rows at all four
outputs byte-for-byte. Python branch trace confirms fused legacy tail versus
separate FP32 pre-mix/20-iteration path. V1 source snapshots archived BEFORE
runtime changes. Component timing is not service timing; small common is slower.

New default-off SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL=1 requires existing
COMMON_FP32=1. A SEPARATE ContextVar/decorator admits only originalV4 TP8/EP1
native eager EXTEND,1..8191 rows,noCP/DCP/PP/TBO/draft/spec/capture. It does not
broaden the mix/post scopes or fake batch metadata. Only MHC dispatch hint is
resolved toNone,using existing FP32/20 reference kernels. Native decode remains
outside this decorator and mode guard. No default/weight/kernel arithmetic edit.

19 CPU tests pass. Integrated-v1(session85164 exit0) covers21 layer/shape cases,
including1/17/63/64/65/127/128/129/255/256/257/2048/2560/8191/8192,three mutations.
Actual selector equals explicit FP32/20 reference and matches large repeated
rows at all four outputs. Report includes actual checkpoint tensor hashes.

Next service intervention is run_service.py in the experiment directory:
baseline small0,max_requests16; mixed small1,max_requests16; serial small1,
max_requests1. All use the EXACT archived historical teacher prompt IDs,route
producerOFF,1Mpool,32Kbudget. No scored performance waves. Compare resulting
1008 logprobs and actual tail scope hits across differing chunk layouts. Do not
require equality to historical mixed FP16/8 math or declare universal batch
invariance. If residual drift remains,locate its next source before promotion.
Reducer and consolidated-launcher GPU screens remain pending.
