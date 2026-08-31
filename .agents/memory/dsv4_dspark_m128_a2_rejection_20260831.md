# DSpark gamma-3 M128 A2 assignment rejection (2026-08-31)

## Scope and safety

- This was a launch-configuration-only screening experiment for the existing
  DSpark gamma-3 `TARGET_VERIFY` path.
- Physical GPUs 0--3, TP4/EP1, BS32, speculative width 4 (`M=128`), original
  weights, 32 real heterogeneous code prompts.
- `SGLANG_DSV4_GFX90A_FP4_GROUPED_DECODE_ASSIGNMENTS=2` selected A2.  No code
  or native-AR default was changed.
- Sparse graph tiers `1 32` were used only to screen the common resident BS32
  window.  Aggregate request wall time is invalid for comparison because
  ingress/egress shapes entered eager/JIT.

## Result

Three resident-BS32 rounds at stream interval 1:

```text
900.880 / 875.145 / 881.829 tok/s
median 881.829 tok/s
```

The retained gamma-3 A4 checkpoint has a recent three-round median of about
`887.837 tok/s`, so global A2 is about `-0.7%` and is not a performance win.
Every round returned 32/32 completions of exactly 256 tokens with
`finish=length`; France first-nine exact and semantic Paris both passed.

## Decision

- Reject A2 as the gamma-3 M128 default and retain A4.
- Do not alter global AIter assignment geometry: doing so could affect native
  AR and other graph tiers.
- Any future occupancy-adaptive experiment must be a strict DSpark-only
  selector requiring `TARGET_VERIFY`, BS32, width 4 and `[128,4096]`, with an
  explicit native-AR negative control even when its environment flag is forced.

