# Current TP8 native decode profile, 2026-09-08

Parent baseline a7f56480af (runtime-M repair retained; direct-X withdrawn).
Re-measured layer20 with existing async, graph-only realtime markers, every16
replays. No weight caches added; pool131072/mem0.80 and graph1/2/4/8/16/24/32
unchanged. Instrumented service2645140 on localhost30011 has been stopped.

C1: 96 complete eight-rank samples, no invalid/incomplete records. France exact,
three measured256-token outputs retain baseline hashes, six supplied-prefix
next IDs/input logprobs/output top20 logprobs exact. After C32, 12 more
fresh-cache transition probes pass the same three exact checks. This time the
diagnostic completed prefill/decode transitions; old instrumentation stalls
did not reproduce. This alone does not establish their historical cause.

C32: two waves of the fixed32 distinct coding prompts (same manifest as the
runtime-M result),256 tokens, native AR. All64 requests completed normally.
32 complete eight-rank samples, no invalid/incomplete records. First logged
sample may be the pending C1 readback; medians use all32 and are not a formal
whole-model latency attribution. Clock factor0.04us/tick follows prior
calibration; raw timestamps from different devices are not subtracted.

| Rank-max median span, us | C1 | C32 |
|---|---:|---:|
|Whole layer20|316.00|832.16|
|Attention MHC/norm|26.40|51.04|
|Attention prepare|102.56|227.68|
|Attention core|29.76|50.64|
|Output projection/collective|39.44|116.00|
|FFN MHC/norm|26.24|51.84|
|MoE/collective|106.24|350.48|
|Router (dual-stream slots16–17)|7.52|27.84|
|TopK (17–18)|15.20|12.80|
|Routed experts (18–19)|50.88|259.04|
|Final collective/arrival (23–24)|23.76|44.32|

Do not sum these independently selected maxima/medians. Markers add launches
and logging changes scheduling; instrumented HTTP rates are not checkpoints.
The previous claimed ~205–220us FFN boundary is NOT current here. C32 prepare
is serial: ~47.28us q-a span,17.60 norm,29.44 q-b,11.36 QK/store,
66.08 indexer branch,58.24 core compressor (includes marker overhead/waits).

Source evidence: attention stream construction admits HIP TP4 only; TP8 M32
cannot use the existing HIP compressor overlap. TP8 BS1 multistream was
previously rejected, but that does not test M32. Next bounded experiment:
default-off TP8/C4/M32/native-decode/unified-KV-only stream gate. Reuse existing
streams, no new weight cache. Do not enable production by inference from this
profile; require E2E and correctness. Pending candidate uses parent2654269,
log `/tmp/dsv4_tp8_m32_attn_overlap_B_20260908.log`.

Artifacts: `/tmp/dsv4_tp8_decode_marker_20260908.log`,
`/tmp/dsv4_tp8_current_{c1,c32}_rankmax_20260908.json` (full ticks),
`/tmp/dsv4_tp8_marker_{c1,c32,transitions}_20260908.json`.
Adjacent JSON contains portable summary. Parser:
`scripts/rocm/summarize_dsv4_realtime_rankmax.py`; C32 starts at log line3072.
