# CK route producer: exact large-M component gain; no runtime integration

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
