# DSpark gamma-three TP-local greedy combination (rejected)

Date: 2026-09-01

## Question

Could the exact TP-local draft vocabulary reduction that retained about 1% on
the older strict gamma-one profile stack with the current gamma-three M128
anchor-only throughput profile?

The candidate used original weights, TP4/EP1/no-A2A, physical GCDs 4--7, and:

```text
SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1
SGLANG_DSPARK_FOLDED_SAMPLING=0
SGLANG_DSPARK_OPT_TP_LOCAL_GREEDY=1
SPECULATIVE_DSPARK_BLOCK_SIZE=3
```

The model-side anchor-only guards remained speculative-only and could not
match native AR. The service captured all requested graph tiers successfully.

## Correctness result

The finite 32-request heterogeneous harness stopped before collecting a
throughput round because the first France sentinel failed:

```text
observed first tokens:
671 6102 294 8760 14 1008 4987 16 270

semantic Paris: false
historical first-nine exact: false
```

The default folded sampler and TP-local sampler have different draft
accumulation/tie behavior. Strict target verification made that harmless in
the earlier gamma-one experiment, but the current gamma-three anchor-only
target approximation consumes the altered non-anchor trajectory and does not
preserve the same quality result.

## Decision

Reject this combination and keep TP-local greedy opt-in. Do not infer that a
proposal-only optimization validated under strict gamma-one is safe under the
approximate gamma-three target path. No performance number is accepted because
the harness stopped at the first correctness gate.

