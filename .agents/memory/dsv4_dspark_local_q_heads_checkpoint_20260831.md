# DSV4 DSpark TP4 local-Q-head checkpoint (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights, TP4/EP1/no-A2A, DSpark gamma one.
- Physical GCDs `HIP_VISIBLE_DEVICES=4,5,6,7`; standalone oracle on GCD 4.
- 32 distinct concrete coding prompts from
  `.agents/memory/dsv4_tp4_code_32_input_ids.json`.
- Four independent services in A/B/B/A order, three 32-request rounds each.
- A retained the legacy H64 padded draft query; B passed only TP4's H16 local
  query to unified-KV attention.

## Why this is valid

The DSV4 target model already passes local Q heads to unified-KV.  The draft
model instead allocated `[T,64,512]`, wrote its 16 TP-local heads into the
prefix, and ran the sparse attention kernel on all 64 heads before slicing the
result back to 16.  Legacy attention backends still require padding and are
unchanged.  The new selector only removes padding when unified-KV is active.

On physical GCD 4, T32 BF16 standalone results were:

| context | padded H64 | local H16 | saving | speedup |
|---:|---:|---:|---:|---:|
| 128 | 105.308 us | 41.845 us | 63.464 us | 2.52x |
| 256 | 135.288 us | 69.811 us | 65.477 us | 1.94x |
| 512 | 197.997 us | 118.059 us | 79.938 us | 1.68x |

All three contexts passed 100 randomized input mutations bitwise against the
first 16 padded heads and 1000 HIP Graph replays with bitwise-stable output.
The graph oracle establishes its reference after the first completed replay;
the capture-time output is not a valid replay oracle when allocator/kernel
initialization occurs during capture.

## Independent-service ABBA

Combined medians across six rounds per arm:

| metric | A: padded H64 | B: local H16 | change |
|---|---:|---:|---:|
| scheduler decode | 778.665 tok/s | 784.158 tok/s | +0.71% |
| host speculative step | 70.330 ms | 70.104 ms | -0.32% |
| common-resident | 792.238 tok/s | 792.530 tok/s | +0.04% |
| aggregate HTTP | 689.765 tok/s | 692.644 tok/s | +0.42% |
| mean accepted length | 1.76382 | 1.76436 | neutral |

One B aggregate round had a non-resident HTTP seam (351.7 tok/s), while its
common-resident throughput remained 792.1 tok/s; the median above intentionally
keeps the predeclared robust aggregation rather than deleting the outlier.

All four services passed the exact France first-nine-token and semantic Paris
oracle.  All 384 coding requests completed 256 tokens with `finish=length`.
Long asynchronous completions retain the already-known cross-service hash
drift; the first 16 tokens and acceptance distributions did not introduce a
new local-head-specific divergence.

## Decision

Enable `SGLANG_DSPARK_GFX90A_LOCAL_Q_HEADS=1` only in the measured TP4 BS32
profile.  Keep the environment default false and preserve the padded path for
legacy backends.  The service gain is below the usual 5% performance-checkpoint
threshold, but the change removes provably redundant work, is graph-stable, and
has a small positive ABBA without changing acceptance.

Artifacts:

```text
/tmp/dsv4_localq_{a1,b1,b2,a2}.json
/tmp/dsv4_localq_{a1,b1,b2,a2}_france.json
scripts/rocm/bench_dsv4_dspark_local_q_heads.py
```
