# DSV4 DSpark confidence calibration and M48 pruning rejection (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights, TP4/EP1/no-A2A.
- Physical GCDs `HIP_VISIBLE_DEVICES=4,5,6,7`.
- 32 distinct concrete coding prompts; greedy generation.
- Static gamma-two/gamma-three acceptance histograms, compact verify-all STS
  collection, then compact gamma-one top-50% verification.

## Per-position acceptance

The request-level DSpark info dumper captured exact `correct_drafts` for real
heterogeneous traffic:

| gamma | samples | correct-draft histogram | mean commit |
|---:|---:|---|---:|
| 2 | 3,879 | 0:977, 1:1,433, 2:1,469 | 2.1268 |
| 3 | 3,628 | 0:939, 1:1,353, 2:727, 3:609 | 2.2773 |

Survival and conditional acceptance were stable across gamma:

```text
position 1: survival 0.741--0.748
position 2: survival 0.368--0.379, conditional 0.497--0.506
position 3: survival 0.168,       conditional 0.456
```

Deeper draft length therefore has rapidly diminishing marginal value. Gamma
one also has a hard maximum of 64 committed tokens per BS32 step, so it cannot
reach 1,500 tok/s at the current roughly 70 ms step even with perfect accept.

## Folded-graph STS collection fixes

The checkpoint includes `mtp.2.confidence_head.proj.weight`, but static mode
intentionally does not construct the confidence head. Compact verify-all was
used to collect confidence without pruning target rows.

Two independent folded-fast-path bugs had made
`SGLANG_DSPARK_STS_COLLECT_PATH` silently produce no shards:

1. folded accept consumes target logits inside `DsparkVerifyEpilogue`, leaving
   `LogitsProcessorOutput.next_token_logits` unset; the observer now reads the
   exact graph-stable `epilogue.strided_logits` view used by accept;
2. `_last_confidence_raw` can retain the capture/warmup batch shape while the
   folded graph output carries the live BS. Collection forbids non-identity
   STS, so the observer reconstructs raw logits with `logit(confidence)` only
   when the shapes differ.

The recorder flush threshold also now counts request samples rather than
Python list entries, and one-time skip reasons replace silent failure.

The fixed production folded path emitted 14 shards / 3,612 samples. France
remained exact and semantic Paris; the diverse 32-request run completed.

Fitted sequential temperatures and ECE:

| position | temperature | ECE before | ECE after |
|---:|---:|---:|---:|
| 1 | 2.2387 | 0.1249 | 0.0529 |
| 2 | 6.3096 | 0.3299 | 0.0881 |
| 3 | 10.0000 | 0.3919 | 0.0897 |

The uncalibrated confidence head is strongly over-confident, but ranking is
useful: survival AUC is 0.726/0.700/0.734 for positions 1/2/3. At position 1,
the top 50% confidence subset has 87.4% precision and retains 58.8% of all
correct first drafts.

## Gamma-one top-50% service rejection

Compact gamma one with `dspark_force_budget_frac=0.5` selected exactly 16
draft rows plus 32 mandatory anchor rows. The observer confirmed 317/317 full
BS32 steps at M48.

```text
aggregate median:       536.50 tok/s
resident median:        648.80 tok/s
scheduler:              662.96 tok/s
host step:               67.94 ms
mean accepted length:     1.446
```

The accepted full gamma-one checkpoint is roughly 779--784 scheduler tok/s,
70.1 ms/step and 1.764 accepted tokens. Removing 16 verify rows saved only
about 2.2 ms because M48 and M64 retain similar expert-union and per-layer
fixed costs, while the removed rows still had high token value. France stayed
exact/Paris and all 64 diverse coding requests completed 256 tokens.

## Decision

- Keep static, verify-all gamma one as the TP4 BS32 default.
- Do not use confidence pruning for the first draft at current M48/M64 costs.
- Keep the confidence data/STS tooling fixes: they are default-inert and make
  future deeper-draft or new-kernel scheduling experiments observable.
- A future adaptive policy must use a target kernel whose cost falls materially
  with selected rows; confidence quality alone cannot overcome the current
  expert-union/fixed-cost floor.

Artifacts:

```text
/tmp/dsv4_gamma{2,3}_accept_hist_server.json
/tmp/dsv4_sts_gamma3_final/sts.*.pt
/tmp/dsv4_sts_gamma3_final/calibration.json
/tmp/dsv4_gamma1_top50_diag{,_france}.json
```
