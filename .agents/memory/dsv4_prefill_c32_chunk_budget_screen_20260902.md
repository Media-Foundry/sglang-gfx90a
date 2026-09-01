# DSV4 heterogeneous C32 prefill chunk-budget screen (2026-09-02)

## Workload

- TP4 / EP1 / no A2A on physical GCDs 4,5,6,7
- 32 distinct code prompts from
  `dsv4_prefill_diverse_32_input_ids.json`
- 73,724 total input tokens, 2303--2304 tokens/request
- original weights, one generated token/request
- 98,304-token pool, 0.96 static-memory fraction
- token-row MHC selector enabled

## 2304 scheduler budget

`chunked_prefill_size=2304` admits one request per scheduler iteration.  Logs
showed `#new-seq: 1` for the whole run.  The completed C32 screen was:

```text
prefill wall: 40.659 s
aggregate input throughput: 1,813.24 tok/s
all 32 requests completed
```

The C1-optimal chunk is therefore a concurrency serialization point; it cannot
serve as the C32 planner policy.

## 16384 scheduler budget

The first attempt exposed an AIter capacity bug: the unregistered custom
reduce-scatter uses a fixed IPC buffer sized for small messages and failed with
`registered buffer is too small to contain the input` on the merged batch.
Token-row collectives now select AIter only when the payload fits its buffer and
shape predicate, otherwise they use RCCL RS/AG.

With that correctness fallback, the scheduler admitted up to eight requests per
iteration (`#new-seq: 8`, `#new-token: 16384`), but the complete run regressed:

```text
prefill wall: 85.342 s
aggregate input throughput: 863.87 tok/s
all 32 requests completed
```

Per-iteration logs varied from roughly 0.83k to 2.53k input tok/s after JIT.
Large-M routed MoE, native owner-local MHC and RCCL collectives outweighed the
reduction in scheduler iterations.  A fixed 16K chunk is rejected.

## Consequence

The next cost-table points should be 4608/6912/9216 (approximately two, three
and four requests per iteration), each after shape prewarm.  Do not infer a
monotonic batching benefit and do not increase directly to 32K/64K.  The 10k
C32 goal remains unmet; current evidence says scheduler batching alone cannot
provide it unless a medium-M kernel region has materially higher useful-token
throughput than both M2304 and M16384.
