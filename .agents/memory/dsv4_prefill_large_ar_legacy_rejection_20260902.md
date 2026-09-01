# DSV4 prefill large-message AIter AR rejection (2026-09-02)

## Component oracle

On physical GCDs 4--7, TP4 BF16 graph replay showed AIter's legacy AR kernel
beating its default new kernel for every tested prefill-sized payload:

| rows x 4096 | bytes | new (us) | legacy (us) | legacy gain |
|---:|---:|---:|---:|---:|
| 1024 | 8 MiB | 330.882 | 294.588 | 11.0% |
| 1536 | 12 MiB | 476.788 | 425.858 | 10.7% |
| 2048 | 16 MiB | 614.425 | 561.311 | 8.6% |
| 2304 | 18 MiB | 678.463 | 629.529 | 7.2% |
| 2560 | 20 MiB | 760.046 | 694.079 | 8.7% |
| 3072 | 24 MiB | 893.243 | 831.809 | 6.9% |
| 4096 | 32 MiB | 1174.977 | 1099.793 | 6.4% |

All outputs were exact and each number is the median of five slowest-rank reps.

## Service integration

A temporary fail-loud size selector retained the new kernel below 8 MiB (thus
leaving decode untouched) and selected legacy at or above 8 MiB. The C32 service
used 32 distinct real code-review prompts, TP4/EP1, no A2A, native checkpoint,
preshuffled AIter FP4 and a 16384-token prefill budget.

Rates were:

```text
2920.75 / 3112.81 / 3108.16 / 3062.64 / 3057.45 input tok/s
median       3062.64
trimmed mean 3076.08
cross-round first-token exact: false
```

The default-new ABBA control was about 3112 input tok/s warm. Therefore the
isolated 6--11% collective improvement did not survive stream overlap and
rank-arrival effects; service throughput regressed about 1%. The selector and
its unit tests were removed.

## Decision

Do not replace large prefill AR with AIter legacy based on the standalone result.
Communication is partially hidden and is not the current structural C32 limit.
Return effort to routed-MoE intermediate traffic and grouped expert execution.
