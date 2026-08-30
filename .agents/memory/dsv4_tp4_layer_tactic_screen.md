# DSV4 TP4 M32 layer-specific A2/A4/A8 tactic screen

Date: 2026-08-30

## Scope

This is an offline, full-routed-stage screen on the real M32 routing recorder
`/tmp/expert_distribution_recorder_1787803355.1855972.pt`, pass 37.  It tests
hash-router layers 0--2, skew representatives 4/16/22/41, and ordinary layer
34.  Every profile includes gate/up, INT8 activation quantization, down, and
the fixed-slot reduction.  Each result is seven GPU-event samples after eight
warmups, with 30 iterations per sample.  Outputs were bitwise exact against
the first profile for every tested layer.

The screen deliberately uses the existing generic grouped oracle before
building any production selector.  Its A4 reference is older than the latest
DPP + down logical-scale path, so a candidate would need to beat this easier
reference by at least 5% before a stricter current-baseline recheck.

## Median full-stage time (microseconds)

| layer | A4 best | A2 R2 B832 | A2 vs A4 | A8 best | A8 vs A4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 461.65 | 477.27 | -3.38% | 525.42 | -13.82% |
| 1 | 474.62 | 484.05 | -1.99% | 543.18 | -14.45% |
| 2 | 478.32 | 483.17 | -1.01% | 536.60 | -12.18% |
| 4 | 446.38 | 452.30 | -1.33% | 491.85 | -10.19% |
| 16 | 424.66 | 443.98 | -4.55% | 486.75 | -14.62% |
| 22 | 434.01 | 468.95 | -8.05% | 490.63 | -13.05% |
| 34 | 437.61 | 474.17 | -8.35% | 517.70 | -18.30% |
| 41 | 446.81 | 482.50 | -7.99% | 521.40 | -16.69% |

`A4 best` is the better of G1664/D832 and G2080/D832. `A8 best` is the
better of R1 B832, R1 B1040, and R2 B832. Negative percentages denote a
slowdown relative to the best A4 result.

## Decision

Reject a layer-specific A2/A8 selector for this recorded M32 pass. Neither
tactic wins on any representative layer, even against the older and therefore
easier A4 baseline. A2 is 1.0--8.4% slower and A8 is 10.2--18.3% slower.
The result agrees with the earlier service-level A2/A8 failures: fewer padded
slots do not repay the extra scan/grid and kernel-shape costs.

Do not expand this table to all 43 layers and do not wire a production tactic
selector. Continue to use A4, with the accepted DPP/down-prefetch/logical-scale
improvements layered on top. Revisit only if the sorter or kernel work
decomposition changes materially, not for another block-count sweep.

Raw log: `/tmp/dsv4_tp4_layer_tactic_screen.log` (host-local, not committed).
