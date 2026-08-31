# DSV4 DSpark TP-local greedy oracle (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights, TP4/EP1, DSpark gamma one.
- Physical `HIP_VISIBLE_DEVICES=4,5,6,7` for every service.
- Resident BS32, 32 distinct concrete code-generation requests, 128 generated
  tokens, greedy decoding.
- France first-nine exact and semantic Paris gates after each runtime change.

The existing TP-sharded Markov-W2 path formed FP32 corrected logits locally and
then all-gathered the complete `[BS, vocab]` tensor before argmax.  The oracle
keeps the same LM-head, BF16 Markov GEMM and `BuildStepLocal` arithmetic, but
selects the first maximum of each contiguous vocabulary shard and exchanges
only one FP32 `(score, global_token_id)` candidate per request.  Rank-order
argmax preserves the full-vector first-token tie break.

The path is opt-in through:

```text
SGLANG_DSPARK_FOLDED_SAMPLING=0
SGLANG_DSPARK_OPT_TP_LOCAL_GREEDY=1
```

It intentionally supports greedy proposal only.  Sampling retains the complete
corrected logits path.

## Correctness

The candidate captured all graph tiers through BS32 and ran without collective
or graph failures.  Its France completion SHA256 exactly matched the default:

```text
3702cfdd7eff2b8f575aeb52e37e1a32bc0ece943f9b672e328e81c6258f56e5
```

The first nine tokens were exact and the decoded answer contained Paris.  All
32 varied code requests completed with 128 tokens and `finish=length`.

Static compilation passed.  The repository unit test could not be collected in
the DS environment because the optional `sgl_kernel` Python module is absent;
the failure occurred during imports before the test body.

## ABBA result

Eight rounds per arm were collected as two four-round services.  Combined
medians:

| profile | aggregate | resident BS32 | scheduler | host step | accept length |
|---|---:|---:|---:|---:|---:|
| default folded sampling | 583.37 | 703.60 | 689.61 | 68.87 ms | 1.589 |
| TP-local greedy | 588.64 | 710.92 | 707.83 | 68.62 ms | 1.588 |

This is approximately +1.0% resident throughput, +2.6% scheduler throughput and
-0.36% host step at matched acceptance.  The gain is real enough to retain as
an oracle, but below the 5% checkpoint threshold and far below the 1.3k target.

Artifacts:

```text
/tmp/dsv4_dspark_foldsampling_a2.json
/tmp/dsv4_dspark_tplocal_a3.json
/tmp/dsv4_dspark_tplocal_b1.json
/tmp/dsv4_dspark_tplocal_b2.json
```

## Decision

Keep the exact TP-local implementation opt-in; do not change the TP4 BS32
default.  The full-vocabulary all-gather is measurable but not a primary
bottleneck because the three-stage draft is only about 8.7% of the complete
gamma-one step.  The next major work must remain in the M64 target verify path,
especially routed FP4 MoE and the attention/MHC boundary.
