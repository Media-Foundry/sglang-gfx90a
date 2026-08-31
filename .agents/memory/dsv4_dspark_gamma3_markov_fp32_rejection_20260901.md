# DSpark gamma-three FP32 Markov-W2 projection (rejected)

Date: 2026-09-01

## Experiment

Test whether restoring the draft Markov-W2 projection from the optimized BF16
path to FP32 improves proposal acceptance enough to offset its extra draft
cost. The candidate used the promoted M128 pre-router compact checkpoint,
original weights, TP4/EP1/no-A2A and physical GCDs 4--7:

```text
SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1
SGLANG_DSPARK_OPT_MARKOV_W2_BF16=0
```

The switch is draft-only and cannot affect native AR.

## Result

The first 32-request heterogeneous round passed the semantic France gate, but
the second round failed before the five-round run could finish:

```text
observed first tokens:
671 6102 294 8760 14 1008 4987 1042 295

semantic Paris: false
historical first-nine exact: false
```

The harness stopped immediately, so no throughput number is accepted. Under
strict target verification a changed proposal remains correctness-neutral;
under the current explicitly approximate anchor-only target, the altered
proposal changes committed hidden/KV trajectories and can amplify across
rounds.

## Decision

Keep the BF16 Markov-W2 optimization. Do not use FP32 as an acceptance-quality
tuning knob for the gamma-three approximate profile.

