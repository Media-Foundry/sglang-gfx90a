# Original V4 TP8 C16: query-group 4 vs 8/16 (2026-09-15)

## Status

Integrated component and 4-vs16 service ABBA completed. Warm throughput is
**6778.9666 -> 6877.4848 input tok/s (+1.4533%)**. Production still defaults
to query-group4: the first candidate wave exposed substantial exact-M JIT
latency, so group16 stays opt-in until runtime-M integration is validated.

Previous goal turn was progress: complete query4 ABBA, +4.5222%, default-profile
selection, raw evidence archive and fork/main push at8d40d5d799. This follow-up
tests only a bounded group-size extension, not another target approximation.

## Exact work decomposition

Reuse the same kernel with BQ8 or BQ16 instead of4. The K tile remains16 keys,
each query still independently executes its64-head dot and fixed head reduction,
and page-ID disagreement retains the per-query fallback. No larger MFMA tile,
shared Top-K, weight precision change, additional output workspace or KV-pool
reduction. The group-size change reduces CTA count and repeated K/scale loads.

The compatibility entry name remains `prefill_query_reuse4`; its new keyword
`query_group_size` defaults to4 and accepts4/8/16. The indexer forwards
`SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE` (fallback4) and logs the actual value
with each rank's hit. No new launcher default has been set. Existing original
V4, native EXTEND, TP8/EP1/CP1/PP1, no-TBO/no-graph, M8192..65536,
C4-width512..2048 guards remain unchanged.

## Component evidence

Only physical GCD4 was used; all GPUs were checked free before experiments.
The control is the accepted **query4 + deterministic logical/physical Top-K**
chain, NOT the earlier nonempty/no-reuse implementation. Synthetic causal8K
requests are used; query producer and compressor are outside timing.

| Candidate | M | Query4 ms | Candidate ms | Speedup |
|---|---:|---:|---:|---:|
| Query8 screen |8192|4.99712|4.33138|1.1537x|
| Query8 screen |32768|19.90771|17.26747|1.1529x|
| Query16 screen |8192|4.97281|4.09091|1.2156x|
| Query16 screen |32768|19.91412|16.41413|1.2132x|
| Query16 strong proxy |8192|4.99690|4.11635|1.2139x|
| Query16 strong proxy |32768|19.93162|16.43312|1.2129x|
| Query16 direct integrated argument |8192|4.99785|4.11598|1.2143x|
| Query16 direct integrated argument |32768|19.92571|16.43964|1.2121x|

Every timing consists of three ABBA cycles, five graph iterations per sample.
One query8 control observation was20.6646ms versus the usual19.90ms; all raw
observations are retained and medians are used, not that outlier as a baseline.
Both group8 and16 report64 registers, no spills and8192 bytes LDS in the initial
launch-proxy screens. Direct-wrapper testing does not separately report compiler
metadata; it uses the same kernel with the integrated group argument.

Initial screens pass10 mutations and10 replays per large shape plus10 ragged
mutations. Both strengthened query16 runs pass100 mutations and100 fixed graph
replays at M8192 and M32768, plus100 M17/width577 mixed-page, ragged-key-tile,
cutoff-tie and strided-page-table checks. All FP32 score bits and logical/physical
Top-K IDs match query4. The direct integrated run, not just launch interception,
is the acceptance gate for going to service.

Driver: `.agents/experiments/dsv4_c16_indexer_qreuse8_20260915/screen.py`.
It reuses the previous integrated fixture. Its outer metadata records
`control_bq=4`, `candidate_bq=8/16`; `direct=true` means it exercises the actual
wrapper keyword, while earlier tests intercept the launch locally. The reused
fixture's legacy bq4 argument does not determine the candidate launch size.
The generic scope string describes the proxy method; the direct flag and actual
driver branch distinguish the final integrated run.

CPU scope, actual rank/group hit-log expression, wrapper default/grid/launch
argument contract, launcher and marker tests: **23 passed,31 subtests**, one
unrelated pytest asyncio_mode configuration warning. The API default is tested
as4, not implicitly changed to16 by integration.

## Service experiment

Directory `.agents/experiments/dsv4_c16_indexer_qgroup16_20260915/`.
Fresh A1 -> B1/B2 -> A2; three timed waves per leg and one excluded warmup per
service. Both arms have the query-reuse family enabled plus all accepted MHC
and empty-tile flags. Only explicit group4 vs16 changes. Actual rank-hit logs
must show the matching group value for all8 ranks.

Original V4, native TP8/EP1/no-A2A, actual1M logical KV, chunk32768,
C16 distinct8K source-code prompts with131069 explicit input IDs, zero prefix
hits, max_new_tokens1. France and two128-output-token quality waves with input
echo/ID-text checks follow independently. The analyzer requires matching source
hashes and timed shape counts; no automatic splicing of a prior control.

Input identity has been verified in preceding runs, but whole-model drift
remains open. Last query4 ABBA had a case5 variant only in the first control's
second wave after57 common tokens. Neither current local exactness nor that
observation alone localizes every projection/CK-atomic/arrival-order difference.

## Completed warm ABBA and quality

| Leg | Input tok/s per wave | Median |
|---|---|---:|
| A1 |6775.986 /6780.516 /6769.239|6775.986|
| B1 |6883.553 /6883.689 /6870.274|6883.553|
| B2 |6871.417 /6880.472 /6869.888|6871.417|
| A2 |6782.232 /6781.947 /6775.844|6781.947|

Mean leg medians **6778.9666 -> 6877.4848 (+1.4533%)**. Mean of leg
median-request TTFT **12.17626 -> 12.00617 s (-1.3969%)**. Whole-wave medians
A1/A2 19.34316/19.32616 s, B1/B2 19.04089/19.07452 s. All four legs have
twelve four-request/M32768 logged forwards, immutable source hashes and
identical input manifests. All eight ranks log the requested group4 or16.

All96 quality input echoes match, zero prefix hits, completion128 and ID/text
consistency pass; all France probes answer Paris. A1 and B each repeat16/16,
and every A1/B quality-wave pairing matches16/16. B also matches the previously
inspected query4 experiment's B first wave16/16 on identical inputs.

A2 repeats15/16. Only case8 differs: A1/B versus A2 first diverges after40
common tokens (2019/666), and A2's own two waves first differ after98 tokens
(21860/108348). The changed excerpts are alternative explanations of distributed
memory helpers, not an observed repetition collapse. These are control-side
variants with exact input IDs; no global deterministic-model claim is made.

All services completed and stopped, with empty owned child lists; AMD-SMI
confirmed all GPUs empty before the follow-up single-GCD experiment.
Evidence archive:88 files, 2,355,942 bytes; SHA256
`4ef392fa3df1247c8afad6b03f906361fb98846b1b57d9f0f445e05012610de5`.

## Cold-shape issue: why group16 is not the default yet

Candidate warmup was3348.396 input tok/s, about39.14s for the131069-token wave,
versus about19.06s warm. Logs explicitly report serving-time compilation of
Triton `reuse` in two bursts, at02:36:05 and02:36:19, with slowest-rank durations
8.77s and9.33s. Do not sum all eight ranks' durations: these are concurrent
rank reports, not144s of sequential service delay.

`M` is a constexpr in both `reuse` and its `emit` helper, although it only
controls bounds. Distinct exact row counts can therefore compile distinct
variants. This is an engineering latency problem independent of the +1.45%
steady-state benefit. Keep group4 as the default while testing a runtime-M
candidate with `do_not_specialize=['M']`; change no score arithmetic.
