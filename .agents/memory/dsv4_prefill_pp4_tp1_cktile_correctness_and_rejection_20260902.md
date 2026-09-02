# DSV4 PP4/TP1 CKTile correctness fix and throughput rejection (2026-09-02)

## Scope

This experiment retested DeepSeek-V4-Flash prefill with four pipeline stages,
one gfx90a GCD per stage, original checkpoint weights, and no expert A2A:

- `TP_SIZE=1`, `PP_SIZE=4`, `EP_SIZE=1`;
- `PP_MAX_MICRO_BATCH_SIZE=1`, `PP_ASYNC_BATCH_DEPTH=4`;
- decode CUDA graph disabled;
- prefill chunk 2304, 98,304-token pool;
- AIter CKTile A16W4 with `KSPLIT=2`.

## Correctness bug and fix

The PP4/TP1 expert shape is W13 `(256,4096,2048)` and packed W2
`(256,4096,1024)`.  Two strict gfx90a DSV4 allowlists omitted that layout:

1. the local CKTile tune installer rejected W13 before dispatch;
2. after enabling dispatch, W2 missed the CDNA2 FlatMM N-lane inverse row
   permutation and produced repetitive wrong text.

Adding only this exact DSV4 layout to both guards restored the semantic oracle:

```text
The capital of France is Paris.
```

No checkpoint weight precision or model mathematics changed.  The fix is a
load-time layout correction already used by the validated TP/EP layouts.

## C32 result

The fixed 32-request heterogeneous code manifest contains 73,724 audited input
tokens per round.  Three rounds produced:

```text
2224.32 / 2238.22 / 2238.00 aggregate input tok/s
median: 2238.00 input tok/s
```

The per-stage logs were around 2.4k input tok/s, but PP scheduling did not turn
the four stages into four-request throughput overlap.  The result is below the
accepted TP4/EP1 ceiling of roughly 2.74k input tok/s.  Cross-round first-token
bitwise equality was also false, although the France semantic oracle was
correct.

## Decision

Keep the narrow CKTile PP4/TP1 correctness support, but reject PP4 as a prefill
throughput profile.  Continue optimization on the TP4/EP1/no-A2A production
path rather than expanding the PP scheduler experiment.
