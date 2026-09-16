# CK route producer: accepted8K/16K service gain;32K regression running

## Latest16K closure, September17

16K A1/B1/B2/A2 medians10132.673885/10214.171757/10219.617908/10136.473024.
Control10134.573454→candidate10216.894832 input tok/s,+0.812283%; control
drift0.037494%, maximum within-leg range0.099763%.192 answers identical and
1008 teacher positions exact logprobs/Top5 against controls and prior K32.
All8 actual route hits,1Mpool unchanged. Prior1376 live checks are8K only;
this regression does not claim additional layerwise diagnostic comparisons.
All three16K services stopped. Same driver session29781 continues into32K;
do not launch the pending single-GCD reducer screen concurrently.
Accepted speeds now8K10372.904520/16K10216.894832/32K9854.324387;32K still
the prior K32 checkpoint pending this trial. Global default unchanged.

## Latest: real8K ABBA accepted, September17

Session53179 exited0; all three owned services stopped and all8GCDs idle.
A1/B1/B2/A2 medians10270.913070/10375.279163/10370.529876/10280.087707.
Control10275.500388→candidate10372.904520 input tok/s,+0.947926%; control
drift0.089326%, largest within-leg range0.432763% (not a statistical CI).
192 full128-token answers identical;1008 teacher positions exact logprobs/Top5
against both controls and prior accepted K32. All8 actual route hits,1Mpool
retained; no decode/spec/precision change. Default remains off. Explicit
launcher and full evidence: dsv4_ck_route_producer_service_20260917/.

At the8K closure,16K/32K still used prior K32 results. Separate drivers in
dsv4_ck_route_producer_regression_20260917/ are gated on8K acceptance; see
latest16K closure above for the updated matrix.

## Earlier component work (historical state below)

Current accepted service matrix after K32: originalV4TP8C16/nativeAR/original
weights/1Mpool,8K10282.763437,16K10151.650740,32K9854.324387 input tok/s.
This new experiment has not changed those E2E numbers.

`.agents/experiments/dsv4_ck_route_producer_20260916/` isolates stage1 output
ownership. Source-hashed CK overlay changes output descriptor and scatter in
both Run and Run_2Lds. Input gather,weight layout,MFMA,v1 geometry,DSV4 bounded
SiLU,splitK1 and BF16 rounding stay the same. Initial CPU replacement-count
test failed because it assumed one method; source inspection found both,then
corrected test passed. No installed AIter/header mutation or production flag.

Two independent clean modules built (token baseline and route candidate).
Source includes distinct CK template names to avoid symbol aliasing. GPU test
compares new baseline against installed stage1 and historical capture first,
then candidate via original Top6-slot inverse mapping. Four mutations incl
expert-block reversal: all stage1 BF16,stage2 FP32partial,finalBF16 exact;
100 graph replays exact. Two real fixtures M8192/M32767. Entire chain includes
stage1,metadata-only identity/inverse,uniqueSetstage2,fixed-order reducer.
No extra BF16 pack. Excludes sorter/dequant from component timing.

| Shape | A eager ms | B eager ms | A graph ms | B graph ms |
|---|---:|---:|---:|---:|
|8192|4.813390|4.818402|4.800192|4.813816|
|32767|17.208379|16.569146|17.155668|16.561548|

Large-M saving0.639ms (~3.71% chain latency); small-M slightly regresses.
Do not broaden to smallM. Assuming43layers×4forwards,isolated saving would
be~0.11s per8K C16 wave (<1%E2E); conditional budget only,not service evidence.
Extra storage relative to production:8,385,536B intermediate+268,337,152B
partial (~264MiB/GCD),plus tiny maps.1Mpool peak not yet validated.

Buildsession20705 andGPU73111 exited0. HIP_VISIBLE_DEVICES5 actually maps
PCI0000:b3:00.0. Service regression had finished and all8 GPUs were idle before
GPU tests. No NUMA/clocks modifications. Build/source/log evidence archived
without object/shared-library bloat;clean rebuild command in README.

Important integration seam: AIter fused_moe.py line1170 reshapes stage1 output
back to(token_num,topk,inter_dim) even with QuantType.No. A simple route-sized
stage1 override cannot work; capacity exceedsM*6. Need explicit same-sorter
two-stage helper or a deliberate ownership-aware boundary. Avoid mutable
global closure/cache coupling. Next useful work is complete routed-stage
oracle including sorter,dequant,allocation plus capacity/layer checks before
serviceABBA. Goal active,not blocked.

## Full routed-stage follow-up (2026-09-17)

`runner.py` explicitly uses the SAME AIter moe_sorting(BM64),then route-stage1,
metadata-only identity/inverse,uniqueSetstage2,fixed reduction. No global
stage-hook mutation or invalid token-major reshape. `full_stage.py` includes
raw FP4 and runtime-scale inverse mapping,direct BF16 expansion into the same
per-device layer workspace,sorting,allocation and complete stage output.
Baseline is the actual production gfx90a_bf16_ck_moe with its accepted flags.
Expanded weights also match the captured shuffled BF16 weights exactly.

First full-stage-v1 failed BEFORE comparisons because the production unique
loader's rank logging needed an initialized TP group. No GPU address/numeric
failure. Its report/log and exact full_stage_v1.py are retained. v2 initializes
a REAL single-rank Gloo SGLang group and cleans it up; standalone scope is
explicitly simulated, not reported as eight-rank validation. v2session80965
and v3session29620 both exited0.

| M | Current full helper ms | Route full helper ms |
|---|---:|---:|
|8192|7.839632|7.769508|
|16384|11.910548|11.578140|
|32767|20.413024|19.860006|
|16383|11.913694|11.600590|
|32765|20.454245|19.875656|
|32768|20.394520|19.828303|
|36864|22.530603|21.854535|

v2 uses two captured real layers/shapes,16384 takes the real prefix. v3 tests
boundaries;32768/36864 append1/4097 real prefix rows to the32767 fixture. These
are component-shape fixtures,not newly captured service batches. Four input
mutations each: hidden/weights,rotated expert IDs,permuted input rows; all full
BF16 outputs byte-exact. Three eager eventABBA cycles per shape. Additional
allocator peak at32767:277604352B (~264.74MiB),no1Mpool service claim yet.

Use16384..36864 for initial service admission: small-M compute-only candidate
was slightly slower even though bypassing generic host dispatch improves the
complete M8192 helper. First integrate only the clearly useful large-M range.
Both arms must retain original-V4 TP8 normal-EXTEND scope; no decode/spec/V4.1.

Suggested narrow integration point: after existing expanded-weight setup and
stage1-kernel selection,but BEFORE installing AIter stage overrides. Explicit
candidate receives expanded weights and uses same sorter,avoiding cached
closure/reshape ownership bugs. A diagnostic reference should run the old path
first under a local ContextVar guard (not toggling global environment),then
candidate,so their large partials need not coexist. Preserve scale contract:
at this point local s13/s2 have already been unshuffled; recursive reference
must receive scales_shuffled=False,otherwise it would inverse-map twice.
Next: opt-in dispatch,unit/source guards,real all-layer exact/capacity oracle,
then C16x8K serviceABBA. No formal speed update from component timings.

## Actual selector integration, September17

Default-off runtime selector now preserves native ordinary-prefill scope and
only admits M16384..36864, I256 and the verified CK contracts. Reference uses
a ContextVar guard and logical scales; no global stage replacement is added.
Five CPU tests and15 scope subtests pass; disabled-path AST is unchanged.

Integrated-v1 failed at JIT because the prototype-relative include was invalid
in the runtime header. The sibling-relative include fixes this; original source,
failure and successful v2 are archived separately. Integrated-v2 session49064
exited0: six shapes, four mutations each, all full outputs byte-exact. M8192
falls back; the other five shapes produce20 exact diagnostic comparisons.
M32768 full-helper timing20.480977→19.877558ms; this is single-GCD component
evidence, not service throughput. Current accepted8K/16K/32K remain
10282.763437/10151.650740/9854.324387 input tok/s.

Next diagnostic driver: dsv4_ck_route_producer_service_20260917/service.py.
Keeps current K32/commonFP32/20/1Mpool/32Kbudget and uses the archived real8K
inputs. Requires all8 actual route hits and1376 full-reference comparisons
before formal ABBA. No new service acceptance yet; default remains off.

Real eight-rank check completed successfully (session62380 exited0):1376 full
routed-output comparisons byte-exact, all8 route hits, France pass and1Mpool
retained. Owned service stopped cleanly. Diagnostic6667.29tok/s duplicates
reference computation and is NOT a performance result. Source hashes frozen.
Formal C16x8K ABBA prepared next; all other accepted switches unchanged.
