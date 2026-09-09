# TP8 DSpark M128 subgroup-8 row-prefetch checkpoint (2026-09-10)

## Scope

- DeepSeek-V4-Flash original checkpoint weights.
- TP8 / EP1 / no A2A, eight MI250 GCDs.
- DSpark gamma three, strict full-target verification (`M=32*(3+1)=128`).
- 32 fixed, heterogeneous real code requests; no repeated-prompt benchmark.
- The selector is default-off globally and is enabled only by the explicit
  TP8 full-target launch profile. Native AR and prefill are ineligible.

## Topology evidence

`amd-smi topology --csv -w/-b` reports all GPU pairs as XGMI, but they are not
equivalent. Same-OAM GCD pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)` have topology
weight 15 and reported bandwidth range 50--200 GB/s. Other direct XGMI edges
have weights 15/30/45 and typically 0--50 or 50--100 GB/s reported ranges.
This matters to TP8 collectives. Expert weights themselves remain rank-local
in HBM, so their row prefetch is an intra-GCD cache/register/LDS optimization,
not a peer-placement optimization.

## Component oracle

The existing gate row-prefetch kernel was paired with the dedicated TP8
subgroup-8 down row-prefetch kernel. The public down helper was not reused: it
is intentionally subgroup-16/TP4 and rejected TP8 with a compile-time
`static_assert`. The TP8 helper binds
`gfx90a_fp4_tp8_down_prefetch_oracle.cuh` explicitly.

Command shape: M128, top-6, E256, H4096, I256, A4/R2/W8, G832/D832. The control
matches the accepted production non-LDS geometry. The candidate uses the
row-prefetch kernels' LDS decode contract.

| Distribution | Control routed stage | Row-prefetch | Improvement |
|---|---:|---:|---:|
| balanced (241 active, max occupancy 8) | 1042.170 us | 753.827 us | 38.251% |
| skewed (189 active, max occupancy 27) | 998.734 us | 706.529 us | 41.358% |

Both cases passed 100 input mutations bitwise exactly against the control and
1000 HIP graph replays bitwise stably. Logs:

- `/tmp/dsv4_tp8_m128_prefetch_balanced_20260910.log`
- `/tmp/dsv4_tp8_m128_prefetch_skewed_20260910.log`

## Real-code service ABBA

Every arm used a fresh process, one excluded warm wave and one measured wave.
Each wave generated 1024 tokens for all 32 requests. The order was control,
candidate, candidate, control.

| Arm | Measured output tok/s | Severe repetition |
|---|---:|---:|
| A1 control | 963.3448 | none |
| B1 row-prefetch | 1081.4361 | none |
| B2 row-prefetch | 1086.2594 | none |
| A2 control | 963.6193 | none |

Means: `963.4821 -> 1083.8478 tok/s`, a **12.4938%** E2E improvement. All 128
measured requests completed the requested length. Artifacts are under
`/tmp/dsv4_tp8_m128_rowprefetch_abba_20260910/`.

## Separate gamma screen

Before this change, a fresh-process gamma-3/gamma-1/gamma-1/gamma-3 ABBA on
2048-token real-code waves measured gamma three at `949.2989 / 966.9513` and
gamma one at `968.0346 / 971.8340 tok/s`: `958.1251 -> 969.9343`, +1.2325%.
Gamma one is retained as a low-latency experiment but does not replace gamma
three: it caps maximum emitted tokens per step at 64 and therefore makes the
2k C32 target require a sub-32-ms step. Artifacts are under
`/tmp/dsv4_tp8_dspark_gamma3_vs_gamma1_abba_retry_20260910/`.

## Decision

Enable M128 row-prefetch in the explicit TP8 full-target profile. Keep the
strict runtime shape/scope checks and do not generalize it to native AR,
prefill, TP4, other speculative widths, or other expert shapes without new
component and E2E validation.
