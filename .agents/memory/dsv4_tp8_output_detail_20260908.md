# TP8 current output-boundary detail

Parent9419d70700. Native TP8/EP1/no-A2A, original weights, pool1048576,
mem0.96, accepted overlap/AR/prefetch configuration retained. Diagnostic only:
layer20 realtime markers, graph-only, sample every16 replays. No additional
weight cache or change to the collective implementation.

Output detail uses slots29(after inverse),30(after wo_a),31(after wo_b matrix
multiply inside the original RowParallelLinear),28(after its complete call).
Do not use slots25/26: MoE prefetch owns and overwrites them. The initial run
used colliding slots and is invalid for output attribution. The new parser
`--output-detail` rejects missing/nonmonotonic markers; rejection was tested
against that initial artifact. No model output corruption was observed.

Corrected run prefix `/tmp/dsv4_tp8_output_detail_v2_20260908`:

| Rank-max median span, us | C1 | C32 workload |
|---|---:|---:|
|Whole layer20|301.60|744.40|
|Inverse-RoPE / entry span|1.60|12.00|
|wo_a|13.12|46.08|
|wo_b matrix multiply|10.40|23.04|
|wo_b collective / arrival|19.20|41.44|
|Output tail|1.60|1.60|

144 complete eight-rank C1 samples,32 C32 samples, zero incomplete/invalid.
C32 logged after one full warm wave, then two32-request256-token waves of the
fixed diverse code workload. Tier is not encoded in the logger; this is the
C32 workload window rather than proof every sample is M32. Each span includes
diagnostic launch/scheduling cost. Do not sum independently selected maxima or
treat12us as removable pure RoPE time. The previous113.6us aggregate came from
fewer markers; these figures are not a new throughput checkpoint.

France exact; all6 measured C1 full256-token outputs match reference;6
teacher-forced next IDs, input logprobs, top20 logprobs match untraced control.
All64 measured C32 requests finish length256; cross-round exact17/32, so full
dynamic-batch determinism is not established. No numerical operation changed.

## Next candidate boundary

TP8 wo_a is one local group, BF16[M32,1,4096] x [1,1024,4096], using einsum
fallback. First compare the actual F.linear/bmm/einsum and a direct GEMM
candidate on real inputs/weights with exact output checks and multiple weight
sets; do not assume a generic CK kernel wins. TP4's two-group stock CK batched
XDL already lost84.624vs38.429us (memory `dsv4_tp4_m32_woa_ck_batched_screen`).
That negative should constrain expectations, but is not an exact TP8 G1 test.
Keep inherited weight precision and memory use; no persistent extra weight copy.

Diagnostic service3097771 remained live after sampling for follow-up; do not
measure acceptance throughput until trace flags are removed and graphs rebuilt.
Raw `_phases.json` gives line ranges; `_c1_rankmax.json`/`_c32_rankmax.json`
contain per-rank ticks; request evidence in corresponding `_c1.json`/`_c32.json`.
