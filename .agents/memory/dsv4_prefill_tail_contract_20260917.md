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

Three-service intervention completed; all three services stopped cleanly.
service-analysis.json validates identical teacher prompts, actual scope hits,
source hashes, 1M pool and different admission layouts. No performance waves.
With the same FP32/20 policy, mixed versus serial still has 544/1008 differing
selected logprobs (max absolute 0.67956644, mean 0.02043068), 15/1008 differing
Top-1 IDs and 381/1008 exactly equal Top-5 records. Requests10..15 match;
requests0..9 differ. This disproves sufficiency of the MHC-tail intervention,
NOT its component correctness. It remains default-off; no new speed accepted.
Next: align logical token rows across small/mixed forwards and identify the
first differing layer/stage. Do not infer MoE, projection, or attention cause
from final logits alone. Preserve this run's source snapshots before probes.

Completed row-aligned first-cached-chunk probes (trace-mixed session32419,
trace-serial session94067; both exit0, services stopped). Added a separate
default-off DEBUG_FIRST_DIV_SMALL admission that requires the existing small
tail policy scope; large optimization scopes remain unchanged.11 CPU tests pass.
Request0 positions32768..32831 match exactly across M32576 mixed / M64 serial;
three layers and all8 ranks captured. At layer0 residual, pre-norm and attention
input are byte-exact on all64 rows. Q differs on every rank (rank0:17/64 rows,
1235/262144 elements,max0.015625). This isolates a difference within Q production
before MoE, not yet its exact projection/norm/RoPE source.

IMPORTANT: trace-analysis.json first_different_stage means observation order,
NOT causal execution order. Existing ffn_mhc_post/comb dumps contain attention
input coefficients generated BEFORE attention but recorded AFTER attn_out.
They also differ already at layer0: rank0 post max0.000177577,comb0.000527471;
all256/1024 coefficient elements differ. Equal BF16 weighted activations do not
prove equal FP32 MHC coefficients. Do not attribute these differences to Q.
Next narrow probes should capture qkv_a,q_lora before/after norm,wq_b before
norm/RoPE, plus first/unfused MHC coefficients at their production boundary.
The current small common policy targets fused post/pre, not every possible
unfused first-layer pre implementation. Do not promote until fully localized.

service-evidence.tar.gz (~8MB) preserves the three original arms and source
versions. trace-evidence.tar.gz (~899MB) is LOCAL-ONLY with tracked manifest:
2874 files including all row hashes/samples/source snapshots. Keep raw files;
do not put this large diagnostic archive in git. trace-values.json quantifies
all64 suffix rows, not a selected few outliers. No new performance score.
Latest accepted P remains8K10372.9045 /16K10216.8948 /32K9854.3244 input tok/s.
