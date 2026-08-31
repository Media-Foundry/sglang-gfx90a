# DSV4 TP4 native-AR BS64 final accepted-bundle A/B/A (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash checkpoint, TP4/EP1/no-A2A, native AR.
- Physical devices `HIP_VISIBLE_DEVICES=4,5,6,7`.
- Graph tiers 1/64 for the controlled return comparison (the initial A service also captured tier 32).
- 64 distinct coding prompts from `.agents/memory/dsv4_tp8_diverse_64_input_ids.json`, 256 generated tokens each.
- Static M64 capacity: 384 routed rows; 65,536-token pool.

## Arms

Arm A is the complete accepted M64 profile. Arm B preserves CK sparse decode,
DPP gate, logical W2 scales, W4 down and LDS unpack, but disables the three
smallest accepted service wins together:

```text
SGLANG_DSV4_GFX90A_M64_GATE_ROW_PREFETCH=0
SGLANG_DSV4_GFX90A_M64_ROUTER_HIPBLASLT=0
SGLANG_DSV4_GFX90A_TP4_M64_C128_ATTN_MULTISTREAM=0
```

The purpose was to catch a possible negative interaction between several
individually small optimizations before closing the environment-switch sweep.

## Correctness

The 64-row fixed teacher-forced oracle compared output IDs, selected-token
logprobs and complete top-5 rows. B and the returned A2 service were 64/64
exact against A1 for all three fields. Every throughput request generated 256
tokens with `finish=length`, and every round retained the exact France prefix.

## Results

Three rounds per service; first rounds after startup are retained but treated
as warm observations when scheduler deltas are unavailable.

| arm | aggregate tok/s | resident tok/s | scheduler tok/s | host step |
|---|---|---|---|---|
| A1, full profile | 907.68 / 903.61 / 912.19 | 888.73 / 882.70 / 890.74 | 1013.82 / 1007.43 / 1016.79 | 63.13 / 63.53 / 62.94 ms |
| B, three weak items off | 864.62 / 896.51 / 873.58 | 838.00 / 870.56 / 848.89 | n/a / 993.54 / 968.80 | n/a / 64.42 / 66.06 ms |
| A2, full profile, graph 1/64 | 879.60 / 904.25 / 910.30 | 857.83 / 887.40 / 889.49 | n/a / 1012.54 / 1015.34 | n/a / 63.21 / 63.03 ms |

The graph-1/64 return arm removes graph-tier selection as an explanation. The
stable A2 rounds reproduce A1 and are roughly 2--4% faster than B, depending
on the metric and B round. The small accepted switches interact positively,
not negatively.

## Decision

Keep all three defaults enabled in the strict M64 profile. Close the existing
environment-switch sweep: the remaining gap from about 1015 scheduler tok/s
to 1300 requires reducing the approximately 63.0 ms step to 49.2 ms, far
beyond any untested configuration switch. Continue only with structural
kernels that reduce real packed-weight traffic and/or a second large
attention/MHC budget.

Artifacts:

- `/tmp/native_bs64_4567_baseline.json`
- `/tmp/native_bs64_bundle_b1.json`
- `/tmp/native_bs64_bundle_a2.json`
- matching `*_teacher.json` files.

