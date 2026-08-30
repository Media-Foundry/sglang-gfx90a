# DSV4 TP4 DSpark full-block and compact-verify rejection

Date: 2026-08-31

## Scope

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- Topology: TP4 / EP1 / no A2A
- Original DeepSeek-V4-Flash checkpoint
- CUDA graph request tiers: 1 and 32
- Workload: 32 distinct concrete coding requests from
  `.agents/memory/dsv4_tp4_code_32_input_ids.json`
- Greedy generation, 256 tokens/request unless noted
- Correctness: the official France completion prefix was checked after every
  runtime configuration change

## Ring-size audit

The DSpark full-block attention stream needs 128 committed SWA rows plus five
draft rows.  The unified pool is already correctly sized to 133 rows:

```text
speculative_num_draft_tokens = 6  # bonus + five draft tokens
spec_extra = 6 - 1 = 5
ring_stride = 128 + 5 = 133
```

The draft graph has `num_tokens_per_req=5`.  Therefore the suspected 132-row
alias did not exist; no ring-size code change was retained.

## Full-block versus causal draft attention

Both services passed the exact France first-nine-token oracle.

| draft attention | mean accept length | resident tok/s | scheduler tok/s | host step |
|---|---:|---:|---:|---:|
| full block, round 1 | 2.374 | 285.41 | unavailable | unavailable |
| full block, round 2 | 2.272 | 299.65 | 323.17 | 180.53 ms |
| causal | 2.361 | 311.05 | 345.34 | 179.06 ms |

Full-block attention did not improve acceptance on the heterogeneous coding
workload and was modestly slower.  Keep `SGLANG_DSPARK_FULL_BLOCK_ATTN=0` by
default.  This is a workload result, not a claim that the official full-block
mathematics are invalid.

## Compact verify forced-budget probes

The same service used `SGLANG_RAGGED_VERIFY_MODE=compact`; the budget fraction
was changed through `/set_internal_state`.  Graph-tier alignment remained on.
Every configuration passed the France oracle.

| forced fraction | tokens/request | mean accept | resident tok/s | scheduler tok/s | host step |
|---:|---:|---:|---:|---:|---:|
| 0.20, first | 128 | 2.447 | 229.44 | unavailable | unavailable |
| 0.33 | 128 | 1.696 | 223.71 | 286.77 | 183.67 ms |
| 0.20, reverse repeat | 256 | 1.921 | 262.48 | 308.30 | 183.36 ms |

Confidence relay, admission seams and graph alignment make one short round
noisy, but the reverse repeat establishes the important result: pruning to a
20% nominal budget does not move the step out of the roughly 183 ms graph
tier.  It sacrifices accepted output without shortening the target+draft
critical path enough to compete with native AR.

## Decision

The corrected DSpark path remains useful as a correctness-complete low-
concurrency research branch, but it is not the BS32 throughput path on four
gfx90a GCDs.  Current native AR is about 650 resident tok/s / 705 scheduler
tok/s, while the strongest DSpark point in this audit is about 345 scheduler
tok/s.  Do not spend the next iteration on SPS fraction tuning.  Return to the
native M32 routed-MoE and FFN-boundary critical path unless a new draft kernel
or graph decomposition demonstrably reduces the approximately 180 ms
speculative step.

## Artifacts

```text
/tmp/dsv4_dspark_ringguard_france.json
/tmp/dsv4_dspark_ringguard_code32.json
/tmp/dsv4_dspark_causal_france.json
/tmp/dsv4_dspark_causal_code32.json
/tmp/dspark_budget_0.20_code32.json
/tmp/dspark_budget_0.33_code32.json
/tmp/dspark_budget_0.20r_code32.json
```
