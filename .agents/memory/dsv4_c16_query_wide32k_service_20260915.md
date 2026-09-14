# C16 x 32K wide-C4 query reuse: completed ABBA, default-off

## Scope and result

Original DeepSeek-V4-Flash, TP8/EP1/no-A2A, native AR; original checkpoint
files, 1,048,576 logical KV slots, chunk/max-prefill 32768. Sixteen distinct
real-code requests, 524,286 actual input tokens/wave, zero prefix-cache hits,
one generated token. This is total input tokens divided by the full prefill
wave time, not decode throughput or per-user speed.

Only `SGLANG_DSV4_C4_PREFILL_QUERY_WIDE` changes: A1=0, B=1, A2=0.
Query group16/runtime-M and the other existing profile flags stay fixed.

| Leg | Three-wave median input tok/s | Median wave seconds |
| --- | ---: | ---: |
| A1 | 4608.316925 | 113.769519 |
| B1 | 5545.886796 | 94.536008 |
| B2 | 5545.714413 | 94.538947 |
| A2 | 4608.043484 | 113.776270 |

Mean of leg medians: **4608.180204 -> 5545.800604 input tok/s,
+20.346869%**. Mean of leg request-TTFT medians:
60.509708 -> 50.303063 seconds, -16.867782%.
All four formal legs have no serving-time compilation warnings; all have
zero cache hits and the required output token counts. This does not claim
the first cold request has the same speed.

All 11 measured source hashes agree across the three fresh processes.
Every candidate rank reports C4 capacity8192/query16/runtime-M1; neither
control reports wide selection. France returns Paris in each process.
All services stopped via owned PID/birth-checked cleanup; the sweep exited0
and `amd-smi process --json` subsequently showed no GPU processes on all8.

## Actual MHC path, not merely configured flags

At this 32K request length each prefill forward admits one request. The
legacy single-request path preempts configured FP32 mix-reuse8. All eight
ranks in every arm report `path=fused_tail batch=1 weight_dtype=torch.float16`.
This is the same preexisting path in A and B; the query experiment does not
introduce it. Do NOT describe this result as executing FP32 pre-mix8.

The scheduler counters are identical: each formal leg logs 48 admissions
of one request with 32768 page-rounded budget tokens. These counters do NOT
prove identical actual model M/row placement. Runtime witnesses include
M32767/32768. See `dsv4_c16_query_wide_20260915/audit_scheduler_counts.py`.

## Input fidelity and drift

All 96 quality-response `prompt_token_ids` exactly equal their manifest IDs;
the manifests themselves are equal across arms and were generated with the
official chat encoding. Both quality waves generate128 tokens per request.

Within-process full-completion repeat counts:

- A1:12/16; B:10/16; A2:9/16.
- First-quality-wave A1/A2:9/16, A1/B:11/16, A2/B:12/16.
- Across all pairings, control/control matches9..10/16;
  A1/B9..11/16 and A2/B8..12/16.
-29/32 candidate excerpts exactly match one of four control waves.
  Candidate-only excerpts: B.0 case13, B.1 case3, B.1 case12.

All candidate texts and unique control alternatives were manually inspected
for coherence, topic and collapse/repetition. No obvious loops or garbling
were found. This is NOT a factual answer-accuracy pass. Case8 sometimes
incorrectly dismisses relevant helper code as missing although the actual
prompt contains those definitions; this variant also appears in the control.
`manual-review.md` records that limitation explicitly. Input equality and
coherence do not prove full-model determinism or target-logit equivalence.

The one-token performance waves separately show case6 varying between
`Looking` and `##` in both control and candidate. The analyzer retains every
leg's first-token vectors and per-case variants. Do not confuse those waves
with the128-token quality runs or attribute their drift solely to query reuse.

## Cold shape and remaining promotion gates

B warmup:4971.015613 input tok/s; first `reuse_runtime_m` compilation took
8.42..8.73 seconds per rank concurrently (do NOT sum these eight times).
A1/A2 warmups:4515.075308/4513.870065. Runtime-M removes exact-row-count
specializations, not the initial new-width/layout compile. Wide-width
prewarming and real mixed-prefix service validation remain outstanding.
**Wide query reuse remains default-off.**

This extends accepted16K evidence (5657.4803 -> 6321.9640, +11.7452%).
It does not replace the accepted8K profile (6959.4558) or establish arbitrary
history widths/concurrencies. No KV history truncation or shared Top-K was used.

## Evidence and reproducibility

Directory: `.agents/experiments/dsv4_c16_query_wide32k_service_20260915/`.
`run.py`, `sweep.py`, `analyze.py`, `review_quality.py`, `manual-review.md`
record setup, metrics and bounded review. The original failed harness run
was separately preserved; it falsely required a mix8 hit and is not part
of this successful ABBA.

Completed archive `evidence.tar.gz`:93 files,7,694,901 bytes,
SHA256 `50263fc855c78735f3799c6c5b79673d643f013045ce427968b5315ff0fd0ce0`.
It contains the full manifests, raw responses, logs, source hashes,
per-process configuration/cleanup records and final analysis.

Next independent work: GPU-screen the prepared large-prefill MHC admission
oracle after service shutdown, keeping small-M AR unchanged; then validate
mixed-prefix wide queries and cold prewarming before default promotion.
