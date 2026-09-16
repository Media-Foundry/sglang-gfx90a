# CK N-stripe exact stage2: scratch benefit, speed regression

2026-09-16, base21e3c6201e. Independent prototype only, no production changes.
Latest accepted C16TP8 speed stays9975.30inputtok/s. Objective was to test the
reviewer's128/256column bounded-scratch suggestion before any service integration.

## Correct contract

Keep current materialized-BF16 CK stage2 V3 instance unchanged: block256,
M64/N128/K64,1x4waves, KBatch1, TypeCastExpertWeight. Preserve sorted original
Top6 slots via existing remap; each assignment writes a unique FP32 Set slot.
Private overlay retains the accepted explicit invalid-store guard.

Changing only N in the CK API would be WRONG: Gridwise computes expert stride
as N*K. Prototype keeps full expert stride4096*256, while stripe N controls the
local output descriptor/grid. B starts at n0*256 BF16 elements; N128-aligned
offsets retain complete preshuffled N64 groups. Full source/header hashes and
build commands are recorded. Installed AIter/CK and production modules untouched.

Stage1 input and sort plan are fixed once. For each stripe: unique CK Set into
[M,6,Nstripe] FP32, fixed Top6 vec4 reduce, store into [M,4096] at global N offset.
One scratch is reused for all stripes. Reducer preserves FP32 addition order
and one BF16 rounding; no atomics, global barriers or publication counters.

## Gates and measurement

Two historical real stage2 captures: M8192 and M32767. Actual inputs, preshuffled
weights, sorted IDs/expert IDs/counts/route weights checked against capture file
hashes. Their upstream model version is historical, not a claim of newly captured
9.975k full-model activations. GPU: HIP_VISIBLE_DEVICES=5, PCI0000:b3:00.0.
No TP8 service overlapped.

Widths128/256/512/1024/2048/4096. For every stripe in each of three input/route-
weight mutations, the ENTIRE FP32 partial is byte-equal to the corresponding
full-width reference slice. Final BF16 output also byte-equal. NaN-poisoned
partial/output and100HIPgraph replays pass.12shape/width cases total.

Full stage2 timing = all stripe producers AND all reducers, comparing against
current full-width unique CK plus current vec4 reducer. Three ABBA cycles with
three complete calls/sample. Eager and graph measured separately. Sort/remap,
stage1, allocations and dequant are outside BOTH timing arms. These are not
full routed-stage or E2E performance numbers.

| Width | M32767 eager baseline ms | Candidate ms | Candidate latency change |
|---:|---:|---:|---:|
|128|10.3683|14.0059|+35.08%|
|256|10.3749|11.9674|+15.35%|
|512|10.4001|11.1358|+7.07%|
|1024|10.3805|10.8747|+4.76%|
|2048|10.3808|10.6613|+2.70%|
|4096 control of prototype overhead|10.3722|10.5801|+2.00%|

Graph results agree in direction; M8192 also loses at every width. The4096
prototype still has its own runtime-stride reducer and compiled CK module;
its2.0% M32767 penalty is implementation overhead, not striping per se. Narrow
128/256 losses substantially exceed it. No unsupported bank-conflict/bandwidth
root-cause attribution: no new hardware counters collected in this screen.

AtM32768 capacity:128columns needs96MiB,256needs192MiB instead of3GiB. Actual
M32767 uses slightly less. **Total logical FP32 partial writes+reads remain
6GiB atM32768**, plus final BF16 output, regardless of stripe width; observed
HBM traffic was not measured. More launches/repeated intermediate accesses and
execution ordering may contribute, but time alone does not isolate each cause.

Decision: keep bounded prototype as a correctness/capacity oracle. Do not connect
it to the service as a speed optimization; do not shrink KV to accommodate a
different candidate. Production unique Set + vec4 stays intact. A cache-local
producer/consumer design would be new work, not merely this smaller buffer.

Artifacts `.agents/experiments/dsv4_ck_nstripe_20260916/`: entry.cu,reducer.cuh,
build.py,screen.py,screen.json, private-build archive/manifest. Building validates
the exact accepted dependency contract and refuses output-directory overwrites.
Installed kernel or checkpoint data never changed. All GPU tests finished and
amd-smi reported no processes on all8GCDs afterward.
