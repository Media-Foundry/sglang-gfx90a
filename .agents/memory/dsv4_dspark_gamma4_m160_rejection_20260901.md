# DeepSeek V4 Flash DSpark gamma4/M160 rejection (2026-09-01)

## Scope

- Hardware: physical GCDs 4,5,6,7; TP4/EP1; original checkpoint weights.
- Workload: 32 concurrent, concrete and varied code prompts, 256 output tokens.
- Runtime: native SGLang single-batch overlap, DSpark block size 4.
- Candidate: target-verify shape `[160,4096]`, five rows per request. Only the anchor row entered routed MoE; four draft rows were masked. The selector required gfx90a, target-verify mode, BS32, width 5, the exact tensor shape, and an explicit environment variable. Native AR could not match it.

## Result

The first complete resident window produced:

- resident BS32: **684.60 tok/s** (3937 tokens over 5.751 s)
- aggregate request wall-time throughput: **401.77 tok/s**
- mean accepted length: **2.408**
- mean acceptance rate: **0.3531**
- all 32 requests: 256 tokens and `finish=length`
- France semantic check: Paris present
- France first-nine token oracle: different from the established reference

The accepted gamma3/M128 checkpoint remains around **0.90k tok/s** on the same workload. Gamma4 therefore regressed resident throughput by roughly 24%, because the larger target-verify graph increased attention/MHC/dense work without enough additional accepted tokens to amortize it.

## Decision

Rejected. The M160 experimental selector and masking implementation were removed. Keep gamma3/M128 as the optimization baseline. This result must not be quoted as an AR result, and no gamma4-specific behavior is allowed to affect native AR.

Raw benchmark: `/tmp/dsv4_gamma4_m160_anchor_allow.json`.
