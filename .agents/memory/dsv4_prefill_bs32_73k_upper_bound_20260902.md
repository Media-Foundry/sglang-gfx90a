# DSV4 TP4 preshuffled BS32 / 73K-token prefill upper bound (2026-09-02)

## Motivation

Earlier large-M measurements were limited to six or eight concurrent 2304-token
requests.  This experiment asks whether one true BS32/M73728 AIter forward gains
enough routed-expert weight reuse to approach the 10k input-token/s objective.

## Two independent admission limits found

The launch script exposed `chunked_prefill_size` but not
`max_prefill_tokens`; the latter remained 16384 and silently capped each
forward at eight requests.  `MAX_PREFILL_TOKENS` is now wired to the CLI.

After raising it, the DSV4 compressor planner rejected more than 65535 ragged
rows because its 16-byte plan stored `ragged_id` in uint16 and WritePlan packed
`batch_id << 16 | ragged_id`.  The plan remains exactly 16 bytes after:

- packing `seq_len` and `buffer_len` as 23+9 bits;
- giving `ragged_id` a full uint32;
- packing WritePlan as 20-bit ragged id + 12-bit batch id.

This supports up to 1,048,575 ragged rows and 4095 requests without increasing
metadata traffic or workspace.  The existing 1M model context fits the bounds.

## Correctness

- All c4/c128/fused consumers compiled against the unchanged 16-byte ABI.
- France returned `The capital of France is **Paris**.` after the change.
- Thirty-two distinct real code-review prompts, total 73,724 input tokens,
  each completed and returned one token.

## Result

The scheduler log proves the intended shape:

```text
Prefill batch, #new-seq: 32, #new-token: 73728
```

Measured throughput:

```text
client last-first TTFT: 2967.53 aggregate input tok/s
server prefill window:  2389.71 input tok/s
```

This is essentially unchanged from the BS8 AIter steady range near 3.0k tok/s.
Therefore current AIter high-M execution does not gain the weight reuse needed
for 10k merely by admitting all requests together.  Larger batching alone is
not the solution; the routed kernel needs a different expert work decomposition
or weight-tile multicast/persistence.

## Related negative oracle

A default-off experiment made the existing small-M grouped sdot kernel read
AIter-preshuffled weights.  M32 was 100/100 mutation bitwise exact, but the
complete routed stage regressed from 1557.61 to 2959.99 us (about 90%).  A16W4
physical order destroys the current lane-wise K coalescing.  That code was
removed; decode remains on its accepted raw/LDS/DPP path.
