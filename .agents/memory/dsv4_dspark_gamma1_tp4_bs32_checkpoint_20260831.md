# DSV4 TP4 BS32 DSpark gamma-1 checkpoint

Date: 2026-08-31

## Scope

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- Original DeepSeek-V4-Flash checkpoint, TP4 / EP1 / no A2A
- 32 distinct concrete coding prompts, 256 generated tokens per request
- Static ragged verify and CUDA graph request tiers 1 through 32
- Independent-service gamma-2 / gamma-1 / gamma-1 / gamma-2 comparison
- Official France first-nine-token and semantic Paris oracle for every service
- Static memory fraction 0.90

## Why gamma 1

At gamma 2 the target pays for a wider speculative verify step and emits an
average 2.13 tokens per request.  Gamma 1 lowers the accepted length to about
1.76, but shortens the host-observed speculative step from roughly 93 ms to
72 ms.  The reduction in target/draft work is larger than the reduction in
accepted output.

Per-round results from adjacent independent services:

| arm | resident tok/s | scheduler tok/s | host step | accept length |
|---|---|---|---|---|
| A1 gamma 2 | 685.749, 698.860 | 657.981, 681.188 | 93.108, 93.355 ms | 2.125, 2.131 |
| B1 gamma 1 | 759.676, 771.380, 773.072 | 753.231, 753.453, 764.719 | 72.759, 71.661, 71.922 ms | 1.752, 1.771, 1.758 |
| B2 gamma 1 | 765.351, 763.640, 767.687 | 753.239, 755.803, 761.246 | 73.000, 72.538, 72.241 ms | 1.771, 1.756, 1.767 |
| A2 gamma 2 | 704.299, 718.489, 725.571 | 706.088, 684.629, 665.773 | 95.681, 88.539, 86.860 ms | 2.146, 2.116, 2.139 |

Median centers over the available samples:

```text
resident:  704.299 -> 766.519 tok/s  (+8.83%)
scheduler: 681.188 -> 754.628 tok/s  (+10.78%)
aggregate: 622.796 -> 679.540 tok/s  (+9.11%)
host step:  93.108 ->  72.389 ms     (-22.25%)
accept:      2.131 ->   1.763        (-17.31%)
```

Every service passed the France oracle.  Every coding request returned exactly
256 completion tokens with `finish=length`.

## Clean default reproduction

The TP4 BS32 DSpark profile was changed to default to gamma 1.  No block-size
override and no `/set_internal_state` forced-budget update were used in the
clean reproduction:

```text
resident:  765.705 / 771.536 / 776.184 tok/s
scheduler: 760.162 / 760.414 / 765.586 tok/s
aggregate: 678.322 / 676.369 / 689.322 tok/s
host step: 72.069 / 72.285 / 71.920 ms
accept:    1.749 / 1.765 / 1.769
France:    exact and semantic Paris
```

Only `SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1` plus an explicitly DSpark command
gets this default.  Other DSpark profiles retain their historical gamma-5
default, and callers can still override `SPECULATIVE_DSPARK_BLOCK_SIZE`.

## Rejected adjacent ideas

The current full-block draft attention implementation matches the official
128-history-plus-complete-block mask and was tested after the CPU sequence-
length contract fix.  It did not improve acceptance and remains default-off.

The existing fused greedy Markov Triton kernel was also screened with the real
checkpoint's BF16 `129280 x 256` W1/W2, BS32 and gamma 2 on physical GCD 4:

```text
eager graph replay:  403.341 us
fused graph replay: 1786.038 us
proposal token match: 96.89% (28/100 mutations fully exact)
```

The rank-256 fused kernel is 4.43 times slower than the hipBLAS-backed eager
path and must stay disabled.

## Artifacts

```text
/tmp/dsv4_dspark_gamma1_{code32,france}.json
/tmp/dsv4_dspark_gamma1_B2_{code32,france}.json
/tmp/dsv4_dspark_gamma2_A2_{code32,france}.json
/tmp/dsv4_dspark_gamma1_default_{code32,france}.json
/tmp/dsv4_dspark_gamma1_{current,B2,default}.log
/tmp/dsv4_dspark_gamma2_A2.log
```
