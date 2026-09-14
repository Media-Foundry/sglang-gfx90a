# Original V4 TP8 C16x16K: wide C4 query reuse service acceptance

Completed fresh-process A1/B1/B2/A2 on original V4 Flash, TP8/EP1/no-A2A,
native AR, original checkpoint precision,1,048,576 logical KV tokens.
Only `SGLANG_DSV4_C4_PREFILL_QUERY_WIDE` changes between control0/candidate1.
Both use query16/runtime-M, pre-mix8, post-fused4 and chunk/max-prefill32768.
No change to the existing8K accepted result (~6959 input tok/s).

## Throughput and latency

16 distinct real public-source code-review requests, each16383..16384 input
tokens,262141 total per wave, zero prefix hits, one output token. Each leg
has3 timed waves after an explicitly separate warmup. Throughput is total
input tokens divided by earliest request start to latest first-token return.
It is not per-user throughput or decode throughput.

| Leg | Three wave input tok/s | Median |
|---|---|---:|
| A1 |5654.7166 /5655.0734 /5654.8707|5654.8707|
| B1 |6326.6799 /6322.6787 /6320.7745|6322.6787|
| B2 |6321.2493 /6324.9873 /6319.5982|6321.2493|
| A2 |5659.8201 /5660.0899 /5660.8285|5660.0899|

Mean of control leg medians5657.480310448183 versus candidate6321.964021214793:
**+11.7452236 percent**. Mean leg-median request TTFT26.1239451 ->23.3996963s
(-10.4281676 percent). Wave medians A1/A2:46.3566741/46.3139288s;
B1/B2:41.4604334/41.4698087s.

All four timed legs have24 forwards of2 requests/M32768 (8 forwards/wave),
zero cached tokens and no serving-time Triton compile warnings. Source hashes
for all measured modules are identical across services. All8 candidate ranks
logged wide-query reuse at rows32768,C4capacity4096,group16,runtime-M1;
controls never logged wide selection. Both MHC implementations also hit on
all ranks. Server info and logs confirm the full1M pool and native AR. All
three services stopped with no remaining owned workers; all GPUs were checked
free before continuing. The user-owned graph-memory pickle was not staged.

## Numerical/output evidence: do not call this global determinism

All96 full input-ID echoes from6 quality waves match their submitted requests;
completion counts, output-ID decoding and texts also agree. All three France
checks answer Paris. Quality waves use128 output tokens at temperature0 with
ignore_eos; they are bounded excerpts, not full answers.

Within-arm complete-output repeat counts: A1=10/16, B=13/16, A2=13/16.
Across independent controls A1/A2, pairwise matches are10..13/16. A/B matches
are also10..13/16. These small samples do not establish equivalent distributions
or identify the cause of every16K drift; they show a substantial control drift
floor despite identical input IDs. Coarse M histograms are equal but do not
prove identical row placement, physical pages or atomic/arrival order.

31 of32 candidate excerpts exactly match at least one of the four control
quality waves for the same request. Only Bwave0/case8 is novel against this
control set. It remains a coherent discussion of `get_available_gpu_memory`,
distributed all-reduce and CPU groups in the supplied excerpt, with wording
differences rather than obvious collapse. Both candidate waves and unique
control differences were manually inspected: no obvious looping/garbled output.
This is not a validation of every asserted bug, model accuracy, or long-generation
stability. The separate component/wrapper oracles establish exact scores and
logical/physical Top-K for their tested fixtures, not global model determinism.

## Cold-shape limitation and default

Warmups: A1=5361.7672,A2=5380.9164,B=5000.9397 input tok/s. B's first wide
`reuse_runtime_m` compilation took8.40..8.63s across the8 ranks, concurrently;
do not sum these rank durations. No such compile warnings occurred in timed
legs. Runtime-M avoids exact-M variant proliferation, but first new width/page
layout variants still require compilation. This is an open cold-prewarm gate.

Keep wider coverage **default-off** while validating32K and real mixed-prefix
behavior. Explicit flag1 admits only the existing original-V4 ordinary-EXTEND
scope, query16/runtime-M, M8192..65536 and C4width<=8192; there is no KV
truncation, precision change or Top-K tie/order change. Native decode, DSpark,
V4.1 and TP4 cannot enter the outer selector. Existing8K behavior is unchanged.

## Reproduction/evidence

Directory `.agents/experiments/dsv4_c16_query_wide_service_20260915/`:
`run.py`, `sweep.py`, `analyze.py`, `review_quality.py`, `package_evidence.py`.
Analysis checks all legs, hashes, actual shape counts,1M pool, hit logs,
full input echoes and decoded outputs before writing `summary.json`.
Review defaults to *not confirmed*; the confirmed artifact records all quality
file SHA256s and candidate-to-control matches. Evidence archive89 files,
4,155,655 bytes; SHA256
`740f0739e7c375bea325306297afcc5b7be6fcc213609cfa33afa74ed1f3b5d2`.

Next: independent C16x32K ABBA using524286 fixed real input tokens, then actual
mixed-prefix cache checks. Neither is measured in this16K result. The32K driver
requires this completed summary and bounded output review before starting.
