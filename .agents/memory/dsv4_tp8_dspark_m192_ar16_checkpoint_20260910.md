# TP8 DSpark strict gamma-five M192 all-reduce checkpoint (2026-09-10)

## Scope

- DeepSeek-V4-Flash original checkpoint weights.
- TP8 / EP1 / DP1 / PP1, strict full-target DSpark, gamma 5, C32.
- The selector is limited to captured BF16 `(192, 4096)` target tensors.
- Native AR and the established gamma-three M128 path are unchanged.

## Component sweep

The two-stage TP8 peer-read all-reduce was compared with its existing
80-block implementation on the 1.5-MiB M192 payload.  Each candidate passed
100 input mutations on all eight ranks and a 32-step, 1000-replay graph chain
bitwise exactly.

| blocks | rank-max median (us) | rank-max trimmed mean (us) |
|---:|---:|---:|
| 80 baseline | 87.60 | 87.63 |
| 12 | 62.68 | 62.75 |
| **16** | **58.63** | **58.76** |
| 24 | 61.87 | 61.83 |
| 32 | 67.40 | 67.60 |

Blocks 16 is 33.1% faster than the paired 80-block component baseline.

## Service validation

Real heterogeneous code prompts, 32 concurrent requests, 1024 generated
tokens per request, two resident rounds:

- 1000.6731 tok/s
- 992.1418 tok/s
- median: **996.4075 tok/s**

The prior M192 routed-MoE + C128 CK checkpoint median was 981.865 tok/s, so
the independent M192 AR selector adds **1.48%** end to end.

Correctness and quality guards:

- France sentinel: 32/32 exact.
- All requests completed 1024 tokens with `finish=length`.
- Tail repeat-8 median: 0.0109 / 0.0129.
- Tail repeat-8 p95: 0.1485 / 0.1089.
- Maximum: 0.3426 / 0.6950; no request exceeded 0.8.
- Logs confirmed all eight ranks hit `target M192 AR ... blocks=16`.

Artifacts:

- `/tmp/dsv4_tp8_m192_ar_blocks{12,16,24,32}.log`
- `/tmp/dsv4_tp8_gamma5_m192_ar16.log`
- `/tmp/dsv4_tp8_gamma5_m192_ar16_france.json`
- `/tmp/dsv4_tp8_gamma5_m192_ar16_c32x1024.json`

## Decision

Accept blocks 16 as the gamma-five strict M192 default.  Keep the environment
selector independent from M128 so neither gamma-three DSpark nor native AR can
silently inherit the M192 geometry.
