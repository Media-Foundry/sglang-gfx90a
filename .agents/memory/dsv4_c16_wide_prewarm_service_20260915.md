# Current long-input service: precompiled wide artifacts hit all8ranks

Tested commit0a0f9e6ecf. Original V4-Flash, TP8/EP1/native AR/no-A2A,
original checkpoint files,1,048,576 logical KV,32768 chunk. Accepted stages1
attention fixedON, optional producerOFF, wide-query explicitlyON. No production
source/default changes. This is coverage/timing observation, not a new ABBA
gain or a full cold-cache service experiment.

## Cache contract verified in the real service

Prime on physicalGCD4 with explicit TRITON_CACHE_DIR=/home/pc/.cache/sglang/triton,
then exit primer before starting TP8. Do not use bare Triton's default as a
proxy for SGLang's configured cache: SGLang redirects third-party caches at
package initialization. Explicit cache path was shared by primer/launcher.
Existing persistent cache was retained, not cleared. Primer calls0.6549s and
0.00355s were cache loads, NOT fresh compiler timing.

France passed before long input. Actual runtime compile-shape logs show all
eight ranks using query16/runtime-M with pointer alignments0,0,0,0,0:

| Workload | Actual M | W / NP / PT | Artifact |
|---|---|---|---|
| C16x16K |32767/32768|4096/64/64|9244e21fabd504689540b2eb80201fffb1cdeacbce5d9cdc767a5e842b7ec37a|
| C16x32K |32767/32768|8192/128/128|0b03280be4d07d3e0abf8698ff50d0b6df4b25e4b8be8e2ed3a40fce607e73f4|

Both match explicit primer artifacts and the prior fresh-cache compiler proof.
Actual stages1 H8/check0 hits also verified on all8ranks in every timing leg.
No compiler markers in these first/warm log windows. Other signatures may
still compile; this does not certify arbitrary contexts or page strides.

## Current observed timing

Reuse frozen public-source manifests and their official chat-encoding hashes:
16K262141 actual input tokens,32K524286. Each length: one first-use wave then
three warm waves, all16 requests/one generated token/zero prefix-cache hits.
128 timing input echoes exact. Analyzer independently recomputes full-wave
rates from earliest begin to latest first-token time; not page-rounded counts.

| Input length | First wave input tok/s | Three warm input tok/s | Warm median |
|---|---:|---|---:|
|16K|6562.039440|6977.511418 /6978.102488 /6979.589766|6978.102488|
|32K|5789.088116|5790.358521 /5790.835165 /5792.643081|5790.835165|

16K first wave39.948099s, warm37.569412/37.566230/37.558225s. Thus~2.38s
first-use overhead remains despite no logged compiler marker. Its cause is
not established here.32K first90.564522s, warm90.544652/90.537200/90.508943s;
32K was tested AFTER16K warmed the process. Do not call it a fresh-process
32K cold-TTFT result. There is no unprimed-service control in this trial.

These are not causal gains over older16K6322/32K5546 measurements: several
accepted optimizations and process conditions changed since those historical
trials. Current default8K production checkpoint remains8384.698465 input tok/s.

## Actual numeric paths and output drift

All8ranks log FP32 mix-pair rows32767/group8/columns2 during16K.32K admits one
request per forward and still hits the legacy fused_tail with batch1 and
weight_dtype=torch.float16, bypassing mix-reuse. This is preexisting behavior,
not a new precision change by priming. Do not describe both lengths as the
same FP32 pre-mix path just because launcher flags agree.

Two128-token quality waves per length:64 exact full input echoes, zero prefix
hits, correct output lengths and tokenizer decoding. Full repeats16K14/16,
32K7/16.55/64 excerpts exactly match prior reviewed long-input replies; all nine
new excerpts were read. No obvious garbling/repetition collapse, but factual
claims are not all validated. Hardware-family and deadlock assertions have
explicit limitations in manual-review.md. This is not an accuracy certificate.

Equal input IDs do not prove identical intermediate KV/row scheduling or
atomic order. Nor can historical-control matching isolate new drift. The next
numerical gate should run real long-input stage1/stage2 per-call comparisons
before blaming attention scheduling. No global determinism claim is made.

## Status and next performance opportunity

Service1865061 completed and stopped with owned remainder=[]; final amd-smi
shows no processes on all8GCDs. All measured source hashes unchanged. Neither
wide-query nor producer defaults promoted. Prewarm is now verified to match
real common16K/32K artifacts, while a strict whole-model cold-TTFT A/B remains
unmeasured. Mixed-prefix current-config coverage remains outstanding.

Code inspection identifies a new bounded opportunity: current query-owner
still rejects W>2048. At16K/32K the full query, logits/Top-K are replicated,
despite existing wide-query reuse. An owner-wide oracle would need to admit
the small owned-row count into the wide operator WITHOUT truncating history,
compare full/owned scores and logical+physical Top-K, and measure complete
eight-rank exchange overhead. Do not simply delete guards or assume8x speedup.
Do not modify the live measured source during a service trial.

Artifacts `.agents/experiments/dsv4_c16_wide_prewarm_service_20260915/`: driver,
raw-time/artifact analyzer, quality comparisons, bounded review, evidence
archive and hash. Source/inputs are frozen; default-off flags are explicit.
Persistent drift/prefill goal remains active.
