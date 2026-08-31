# DSpark gamma-three partial-accept shadow analysis (rejected)

Date: 2026-09-01

## Candidate

Evaluate a DSpark-only approximate accept policy without changing runtime
commits. Row zero would still use the target logit to verify draft zero. Later
drafts would be trusted only when the existing calibrated confidence prefix
exceeded per-position thresholds. Native AR cannot enter this proposed path.

## Data

The shadow analysis used all 14 strict-target STS shards in
`/tmp/dsv4_sts_gamma3_final`, totaling 3,612 samples from heterogeneous
requests. The fitted sequential temperatures were 2.2387, 6.3096 and 10.0.
Actual target survival was:

```text
draft 0: 74.3079%
draft 1: 37.2093%
draft 2: 16.2791%
```

## Hard throughput bound

When draft zero is rejected, this policy must emit the strict target bonus and
cannot trust a later prefix. With gamma three, even trusting both remaining
drafts whenever draft zero matches gives:

```text
maximum mean commit = 1 + 3 * P(draft0 correct)
                    = 1 + 3 * 0.743079
                    = 3.22924 tokens/step
```

The current target-step time needs roughly 3.4 committed tokens per step to
reach 1.5k tok/s. Therefore this policy cannot reach the goal even with zero
thresholds and no implementation overhead.

## Precision/throughput tradeoff

A 0.005-spaced exhaustive threshold grid over the calibrated cumulative
confidence produced:

| constraint | best mean commit | trusted-token precision |
|---|---:|---:|
| precision >=99.9% | 1.7494 | 100% over only 23 trusted decisions |
| precision >=95% | 1.7517 | 96.77% |
| precision >=90% | 1.7805 | 90.37% |
| mean commit >=3.0 | 3.0091 | 39.84% |
| mean commit >=3.2 | 3.2046 | 36.43% |

No threshold pair reached mean commit 3.4. The high-throughput region would
commit mostly incorrect unverified prefixes and would compound the existing
anchor-only hidden/KV approximation across rounds.

## Decision

Do not implement gamma-three confidence partial acceptance. It is both below
the mathematical throughput requirement and far below the quality gate. A
gamma-five variant has a different upper bound but requires new five-position
confidence calibration and already inherits a known cross-round repetition
risk; it must not be inferred safe from this analysis.

