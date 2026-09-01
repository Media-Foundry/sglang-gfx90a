# DSV4 TP4 prefill M2304 and C32 delayer profile (2026-09-02)

## Workload and invariants

- Physical gfx90a GCDs 4--7, TP4/EP1, original checkpoint weights.
- Thirty-two distinct real code prompts from
  `.agents/memory/dsv4_prefill_diverse_32_input_ids.json`.
- Each prompt contains 2303--2304 server-audited tokens.
- Concurrent prefill measurements generated one token so completed requests
  immediately released KV and did not trigger retraction/re-prefill.
- France correctness returned `The capital of France is Paris.` on both tested
  weight-layout paths.

## Strict global-2304 baseline

With `chunked_prefill_size=2304`, every scheduler forward admitted one request.

- C1 steady median: 2356.2 input tok/s for the 2304-token manifest request.
- C32 steady median: 2357.6 aggregate input tok/s.

The standard 4604-token C1 prompt remains faster per token (about 2.49k input
tok/s) because it amortizes fixed request and final-head costs over two near-full
chunks. Both lengths must be reported; the longer prompt cannot serve as the
only C1 oracle.

## M2304 direct-path profile

The TP0 EXTEND trace contained about 963.85 ms of GPU kernels. Forty-three-layer
totals were:

| component | total ms |
|---|---:|
| raw-FP4 MFMA64 gate/up | 326.00 |
| raw-FP4 MFMA64 down | 234.83 |
| MHC pre-mix | 88.38 |
| TP collectives | 73.86 |
| sparse prefill attention | 39.39 |
| MHC post combine | 31.18 |
| full indexer logits | 23.65 |
| both group32 INT8 quant sites | 3.97 |

Gate/up plus down account for about 58% of kernel time. The standalone quant
launches are too small to explain a large end-to-end gain; a gate epilogue
fusion is useful only if eliminating BF16 intermediate traffic also shortens the
producer/consumer kernels materially.

## Five-millisecond cold-burst delayer

The existing prefill delayer was configured with:

```text
chunked_prefill_size=16384
prefill_max_requests=7
prefill_delayer_queue_min_ratio=1
prefill_delayer_max_delay_ms=5
prefill_delayer_max_delay_passes=10000
```

It successfully changed the real service batch distribution. Across three C32
rounds, most forwards contained six or seven requests:

```text
11 x (6 requests, 13824 tokens)
 3 x (7 requests, 16128 tokens)
```

Therefore HTTP arrival skew and immediate first-request issue were real, and the
delayer is a valid mechanism for reaching the large-M oracle. It is not itself
the compute solution.

## AIter/preshuffled large-M result

With the large-M AIter path, C32 rounds were 3018.8, 3159.6, and 3068.5 input
tok/s (median 3068.5). C1 fell to a median of 1875.1 input tok/s, so this cannot
replace the accepted latency/direct layout.

The first profiler attempt without a delayer captured only `bs=1,toks=2304`,
which proves that a large global token budget alone does not create a large
forward. Profiles must verify the step annotation before being interpreted.

## Raw/direct M16128 result

The delayer produced a verified `step[EXTEND bs=7 toks=16128]`. Its TP0 profile
contained:

| component | total ms |
|---|---:|
| raw-FP4 MFMA64 gate/up | 2075.71 |
| raw-FP4 MFMA64 down | 1968.60 |
| MHC pre-mix | 639.90 |
| MHC post combine | 310.82 |
| NCCL collectives | 302.27 |
| sparse prefill attention | 258.78 |
| down reduction | 57.67 |
| both INT8 quant sites | 26.20 |
| all GPU kernels | 6630.28 |

After the M16128 JIT shape was warm, a full C32 service round still reached only
1051.4 aggregate input tok/s. The direct kernel is therefore a latency layout,
not a high-occupancy production path.

## Next kernel direction

At M16128, the average routed occupancy is roughly 378 assignments/expert.
MFMA64 consequently scans each expert weight about six times. Do not increase a
single wave's accumulator from 64 to 96/128 rows; that route already loses
occupancy. The next oracle should keep 64-row registers per wave group while two
assignment groups share one LDS-staged weight tile:

```text
assignment group 0: 64 rows, K split 2
assignment group 1: 64 rows, K split 2
four waves/CTA total
one shared raw-FP4 weight tile per K phase
separate fixed-order partial reductions
```

This preserves the current per-wave accumulator footprint and targets the
dominant repeated weight scans. It must first beat the complete gate/down stage
at M13824/M16128 by at least 20%, remain exact, and leave the M<=2304 selector
unchanged. Even a 2x routed-stage result would not alone prove the 10k service
goal; full-service ABBA remains mandatory.

