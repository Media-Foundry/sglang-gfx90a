# DSV4 breadth-one NGRAM correctness support and throughput rejection

Date: 2026-08-31

## Scope

- Original DeepSeek-V4-Flash safetensors, TP4/EP1/no-A2A.
- Physical GCDs `4,5,6,7`; helper oracle on physical GPU 4.
- NGRAM was restricted to `min_bfs=max_bfs=1`. General trees remain rejected
  because DSV4 HIP verify does not consume the tree mask and compressor state
  is not branch-local.
- Correctness gates were the France first-nine/Paris oracle, 256 generated
  tokens crossing the C128=128 boundary, and 32 distinct coding prompts.

## Commit-path correction

For a topk-one chain, `assign_extend_cache_locs_func` writes the target verify
inputs directly into the contiguous
`req_to_token[seq_len:seq_len + draft_width]` window. The accepted path is a
prefix of the same window, so the generic post-verify slot move is an identity.
`DeepSeekV4TokenToKVPool` now advertises only this narrow contract; NGRAM skips
the unavailable generic mover for that pool and fails loudly if topk is not
one.

NGRAM also now calls `clear_unaccepted_c128_draft_states` before commit, matching
the EAGLE-family DSV4 verify path. Without this, rejected candidates could leave
stale C128 compressor state.

The direct helper oracle verified:

- the C128 cleanup receives request slots, pre-verify sequence lengths,
  accepted lengths and draft width;
- the generic mover is not called for the identity chain;
- a tree/topk-two request raises immediately.

The repository unit-test module could not be collected in this environment
because the optional `datasets` package is absent. The same assertions were run
directly, followed by the GPU E2E checks below.

## NGRAM3 without external corpus

France generated all 256 tokens, crossed position 128, and passed first-nine
exact plus semantic Paris. Mean accepted length was `2.246` for that single
request.

On 32 distinct code prompts, the corpus warms across rounds, so only the first
round is a clean unseen-request measurement:

| round | aggregate | resident | scheduler | host step | mean accepted |
|---:|---:|---:|---:|---:|---:|
| 0 | 402.80 | 442.88 | 431.21 | 84.92 ms | 1.225 |
| 1 | 487.44 | 605.63 | 535.13 | 85.34 ms | 1.603 |
| 2 | 518.24 | 706.21 | 549.61 | 77.26 ms | 1.801 |

The later increase is repeated-request memorization and is not admissible as a
general throughput claim. It still did not beat DSpark gamma one.

## Independent external-code corpus

A separate JSONL corpus was generated from 215 tracked SGLang source files,
excluding benchmark generations: 1,999,996 characters and 531,563 model
tokens. The narrow valid configuration was NGRAM2 with one external suffix
automaton slot.

France again completed 256 tokens and passed the semantic oracle, but accepted
only `1.020` tokens/step. On the first round of the 32 unseen code prompts:

```text
aggregate:             449.87 tok/s
resident:              485.35 tok/s
scheduler:             485.84 tok/s
host step:              68.73 ms
mean accepted length:    1.049
```

All 32 requests completed 256 tokens with `finish=length`.

## Decision

Keep breadth-one DSV4 NGRAM as an explicit experimental/correctness-supported
profile, but do not enable it by default and do not use it for the 1.5k goal.
Both cold online-corpus NGRAM3 and independently preloaded NGRAM2 are far below
the accepted DSpark gamma-one range (~770--784 scheduler, ~765--790 resident).
General tree NGRAM remains unsupported and fail-loud.

Artifacts:

```text
/tmp/dsv4_ngram3_linear_france256.json
/tmp/dsv4_ngram3_linear_code32_r3.json
/tmp/dsv4_ngram2_corpus_france256.json
/tmp/dsv4_ngram2_corpus_code32_r1.json
/tmp/sglang_code_corpus_2m.jsonl
```
