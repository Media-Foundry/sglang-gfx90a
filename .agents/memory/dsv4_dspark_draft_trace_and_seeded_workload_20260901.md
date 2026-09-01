# DSV4 TP4 DSpark draft trace and seeded-workload checkpoint (2026-09-01)

## Scope

- Original DeepSeek-V4-Flash checkpoint weights.
- Physical `HIP_VISIBLE_DEVICES=4,5,6,7`, TP4/EP1/no-A2A.
- DSpark gamma 3, graph tiers 1--32, 49,152-token pool.
- Thirty-two heterogeneous requests, `stream_interval=1`.
- Every runtime screen checked 1,024 generated tokens/request, `finish=length`,
  and the France semantic-Paris sentinel unless explicitly noted as a trace-only
  short run.

## Reproducible heterogeneous workload

`scripts/rocm/bench_dsv4_tp4_diverse_concurrent.py` now accepts
`--request-seed` and `--workload-output`. A seed samples/orders the concrete
request pool once, pins the France request at slot zero, and reuses exactly the
same `input_ids` in every round. The JSON report records
`selected_workload_sha256`; a materialized workload can be passed back through
`--inputs` for an A--B--B--A service comparison. `cache_salt` remains unique per
round so prefix-cache reuse cannot contaminate timings.

This is the required comparison protocol going forward: randomness is between
experiments, never between arms of one experiment.

## Draft graph decomposition

The default-off `SGLANG_DSPARK_GFX90A_REALTIME_TRACE_STAGE` selector extends
the existing gfx90a realtime marker through a DSpark stage and the folded
proposal tail. It is DSpark-only; native AR and normal DSpark launch no marker.

At the full resident BS32/gamma3 tier, grouping four ranks and taking rank-max
gave these medians over nine full-tier samples:

| Region | Time |
|---|---:|
| complete marked draft graph | 8.228 ms |
| stage 0 | 1.867 ms |
| stage-0 attention | 0.404 ms |
| stage-0 routed/shared MoE region | 1.262 ms |
| remainder of stages 0--2 | 3.718 ms |
| HC collapse + local LM head | 0.480 ms |
| three-step Markov proposal | 2.154 ms |
| confidence tail | 0.004 ms |

The independent DSpark segment timer measured roughly 10.0 ms for the whole
draft interval, leaving about 1.8 ms in embedding, graph boundary, and work
outside the marker span. The target verify interval remains roughly 58.5 ms,
so the target is still the dominant 2k barrier.

Raw artifacts:

```text
/tmp/dsv4_2k_draft_tail_trace_screen.json
/tmp/sglang_dsv4_flash_dspark.log
```

Marker kernels perturb graph scheduling and are diagnostic only. The France
sentinel remained semantic in the final trace run, but trace throughput is not
a checkpoint.

## Greedy-only folded proposal rejection

The default folded sampling graph generates full `[BS,vocab]` exponential
noise even when all requests are greedy. Gamma3 greedy-only retained the same
full-vocabulary Markov-W2 gather and argmax order but disabled sampling buffers
and RNG:

```text
SGLANG_DSPARK_FOLDED_SAMPLING=0
SGLANG_DSPARK_OPT_TP_LOCAL_GREEDY=0
```

Five 32x1024 heterogeneous rounds produced resident BS32:

```text
1582.92 / 1522.77 / 1528.96 / 1555.42 / 1541.89 tok/s
trimmed center: 1542.09 tok/s
```

All five passed semantic Paris and all requests returned 1,024 tokens with
`finish=length`. This is effectively identical to the accepted AR12 + M32 row
prefetch checkpoint center (about 1541 tok/s in its three candidate rounds),
so removing the logical RNG work does not survive graph scheduling. Keep
folded sampling enabled by default and do not combine this rejection with the
separately rejected TP-local greedy path.

Artifacts:

```text
/tmp/dsv4_2k_gamma3_greedy_only_screen_1024_r2.json
/tmp/dsv4_2k_gamma3_greedy_only_1024_r3.json
```

## Next priority

Do not spend the 2k effort on Markov-W2 or the confidence tail. The full draft
transformer stack is about 5.6 ms and the target verify is about 58.5 ms. A
credible next checkpoint must remove multiple milliseconds from target
attention/MHC/MoE or overlap the draft interval with target work; sub-0.5-ms
tail changes cannot close the roughly 7-ms perfect-acceptance step gap.
