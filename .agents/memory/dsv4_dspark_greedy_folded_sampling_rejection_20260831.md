# DSV4 DSpark greedy-only folded-sampling rejection (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights.
- TP4/EP1, DSpark gamma one, resident BS32.
- Physical GCDs were `HIP_VISIBLE_DEVICES=4,5,6,7` for every service.
- Workload was 32 distinct, concrete code-generation prompts, 128 generated
  tokens per request, greedy decoding, and `stream_interval=1`.
- The France oracle was run after each service change.

The hypothesis was that `SGLANG_DSPARK_FOLDED_SAMPLING=0` would avoid the
sampling-only exponential-noise and corrected-logit work while retaining the
folded greedy proposal in the draft CUDA graph.  The B service did report:

```text
DSpark draft proposal (greedy only) folded into the draft cuda graph.
```

The default A service reported `greedy + sampling`.

## Correctness

Both A and B produced the same France completion SHA256:

```text
3702cfdd7eff2b8f575aeb52e37e1a32bc0ece943f9b672e328e81c6258f56e5
```

Both passed the exact first-nine-token oracle and the semantic Paris check.
All 32 code requests completed 128 tokens with `finish=length`.

## ABBA result

The most comparable B/A2 rounds had almost identical mean accepted length:

| profile | accepted length | host step | scheduler tok/s | resident tok/s |
|---|---:|---:|---:|---:|
| B greedy-only, four-round median | 1.587 | 70.41 ms | 683.14 | 691.10 |
| A2 default, four-round median | 1.589 | 68.48 ms | 689.61 | 705.50 |

The first A service also measured 68.13 ms at accepted length 1.589.  Thus the
greedy-only folded path regressed host step time by about 2.8% at matched
acceptance.  Aggregate throughput moved with acceptance and is not used as the
primary comparison.

Raw outputs:

```text
/tmp/dsv4_dspark_foldsampling_a1.json
/tmp/dsv4_dspark_foldsampling_b.json
/tmp/dsv4_dspark_foldsampling_a2.json
```

## Decision

Reject `SGLANG_DSPARK_FOLDED_SAMPLING=0` for the TP4 BS32 profile and retain the
default folded `greedy + sampling` implementation.  The likely reason is that
the generic folded path has better graph/kernel scheduling despite doing more
logical work.  Do not infer a benefit from removing corrected-logit buffers
without an end-to-end graph measurement.

The next higher-value draft-tail experiment is TP-local corrected top-1:
compute each rank's local `(value, token_id)` winner after applying its local
Markov bias, exchange only the four winners, and preserve global token-id
tie-breaking.  This can remove the full `[BS, vocab]` all-gather without
resurrecting the previously rejected fused Markov-W2 Triton kernel.
