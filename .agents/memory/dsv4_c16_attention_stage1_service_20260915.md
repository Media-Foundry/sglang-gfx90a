# Accepted native TP8 C16 prefill attention pipeline: 8.385k input tok/s

Active drift/prefill goal progressed, not globally completed. Original V4-Flash,
TP8/EP1/no-A2A/native AR, original checkpoint weights,1M logical KV pool,
32768 chunk,16 real heterogeneous code prompts,131069 input tokens per wave.
**Query producer remains OFF in both arms**: no new rocBLAS projection numerical
path, no shared indices, no skipped experts or KV, no changed K16 softmax grouping.

## Implementation and scope

`SGLANG_DSV4_PREFILL_ATTN_STAGE1` selects `num_stages=1` for existing two-bank
prefill attention. The kernel body is unchanged. Optional keyword passes through
runtime/public wrapper to the Triton launch. Omitted keyword preserves compiler
default; OPUS rejects a Triton-only override. No process-global compiler mutation.

Pure policy plus backend arguments require original model_type deepseek_v4,
gfx90a, ordinary native EXTEND, no speculative algorithm/draft, TP8,
CP/DCP/PP1, BF16 H8/D512 and M8192..65536, outside graph capture. Decode returns
before the prefill branch. Actual backend model type is saved at construction.
Small/nonmatching paths retain original behavior. C1 France passed in all fresh
services before any large-prefill hit; this is not a new C1 speed measurement.

The preexisting component record proved stages1 vs stages2 byte equality on
ragged/empty/sentinel/duplicate-index fixtures, non-contiguous strides,
M1/17/129/8191/8192/32767/32768/65536,100 mutations and1000 graph replays atM129.
A runtime-wrapper GPU smoke check atM17 also passed byte equality.

Independent fresh diagnostic service1803712 enabled
`SGLANG_DSV4_DEBUG_PREFILL_ATTN_STAGE1_CHECK=1`: every admitted call computes
both original and candidate attention and compares output bytes. The16 actual
code requests/128 output tokens completed, all input echoes/cache-zero checked,
all eight ranks hit the path withcheck1, no mismatch. This is same-live-input
local operator equality, not full-model global determinism. Diagnostic stopped
before timing services and is excluded from throughput numbers.

## Formal ABBA

Fresh A1/B/A2 services; B supplies consecutive B1/B2 legs. Three waves per leg,
one excluded warmup per process. Frozen source/input manifests match across
timing arms; the sole candidate flag change is pipeline stages. No GPU probes
or competing benchmarks ran alongside service. Diagnostics disabled for ABBA.

Throughput recomputed from raw independent streaming-client timestamps:
131069 divided by earliest request begin to latest first-token arrival.
All192 formal input echoes exact, all prefix-cache hits zero, one output token
per request. All eight ranks hit owner mode in both arms and stage1 in B only.

| Leg | Three input tok/s measurements | Median input tok/s | Median wave seconds |
|---|---|---:|---:|
| A1 |7958.070 /7958.542 /7956.644|7958.069835|16.469948|
| B1 |8378.237 /8381.594 /8391.111|8381.593884|15.637718|
| B2 |8387.803 /8397.535 /8370.747|8387.803047|15.626142|
| A2 |7965.775 /7966.085 /7960.125|7965.775425|16.454016|

Mean of leg medians: control **7961.922630**, candidate **8384.698465 input
tok/s**, **+5.309972%**. Return control reproduces; candidate gain exceeds wave
spread. Request TTFT medians A1/B1/B2/A2:10.478509/9.955315/9.945938/10.461627s,
not to be confused with whole-wave duration. No compile markers in formal logs.

Observed owner signatures M32766/32767/32768, localM3072,width2048 in all legs;
B2 additionally hasM32765. Stage1 logs have the same actual largeM/H8/check0.
Identical request tokens do NOT imply identical internal batch-row ordering;
this minor admission-shape variation is explicitly retained in evidence.

## Output/precision verification

Two additional128-token quality waves per service:96 exact input echoes,
zero cache, validated completion lengths and tokenizer.decode==response text.
B repeat16/16; A1 repeat15/16, A2 repeat15/16, independent controls15/16.
Both candidate waves match A1.0 on all16 complete outputs (32/32); no new text
branch or collapse relative to that control. Generated code-analysis claims
were not newly certified as factual; equality to control is a numerical witness.

This does not erase known full-model/CK atomic reduction drift. Some control
repeats still diverge. We have isolated exact attention schedule substitution
at fixed live inputs and found no added output drift in this trial, not solved
all possible dynamic batching or FP32 atomic nondeterminism.

## Acceptance and defaults

Promote only inside the combined TP8 multi-request + prefill-throughput launcher
profile, with existing TP8/EP1/no-A2A outer guard. Explicit flag0 overrides it;
other launch profiles remain unchanged. Runtime gates above still apply.
`SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER` remains default-off; its prior8.237k
non-bitwise result is not stacked here or relabeled as this checkpoint.

After ABBA, only launcher defaults changed among hashed runtime sources.
33 CPU scope/flag/launcher-prefix tests passed (one preexisting pytest config
warning: unknown asyncio_mode), bash syntax and diff whitespace checks passed.
Acceptance JSON records tested/current launcher hashes. Runtime sources used
in the service are unchanged by promotion. No published claim that every
16K/32K/mixed-prefix workload benefits; those remain coverage work.

All four owned services stopped; amd-smi reported no running process on any
of the eight GCDs. User-owned memory-usage pickle and unrelated files preserved.

Artifacts: `.agents/experiments/dsv4_c16_attention_stage1_service_20260915/`:
run.py/sweep.py/analyze.py/review.py, flag and launcher tests, acceptance.json,
and archived raw A1/B/A2/check plans, input/response IDs, timestamps, service
logs, stop records, analysis and quality comparisons. See evidence.sha256 for
archive identity. Current accepted prefill speed is now **8.385k**, not7.954k.

Next: quantify new remaining attention budget and consider testing this exact
improvement with wider lengths/prefixes or stacking the optional producer,
keeping the latter's documented numerical tradeoff separate. Do not count the
old2.76s attention budget as wholly unoptimized after this promotion.
