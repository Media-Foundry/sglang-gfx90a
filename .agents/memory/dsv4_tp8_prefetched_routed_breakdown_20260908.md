# Accepted-prefetch TP8 routed chain: refreshed isolated budget

## Scope

2026-09-08, baseline HEAD 48da17a8ae. No production kernel or selector changed.
Existing TP8 service retained the 1M token pool. AMD-SMI ownership check found
only the baseline service tree before running the standalone probe on GPU4.
The first invocation failed before GPU work because PYTHONPATH omitted the
repository python directory; corrected invocation completed successfully.

Added `--breakdown` to `scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py`.
It retains the current TP8 I256 A4/R2/W8/B832 geometry and prefetched gate.
The down-prefetch experiment remains disabled. Fixed-slot arithmetic unchanged.

This is synthetic input/weight/routing screening, not a captured decode trace.
Gate scale metadata is mutated every25 iterations, activation/input scales
every iteration; down scales vary by element. Three expert pools were tested.
Each pool passed100/100 intermediate, FP32 partial and final-output comparisons;
repeated graph output was stable. The separately captured chain also exactly
matched the full graph's partial and final output.

## Trimmed microseconds

| Active experts / A4 scans | Prefetched chain | Gate | Intermediate quant | Down partial | Fixed reducer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 133 / 133 | 275.037 | 148.018 | 7.252 | 122.404 | 6.068 |
| 104 / 105 | 241.906 | 130.575 | 7.313 | 108.994 | 6.106 |
| 32 / 61 | 183.184 | 99.553 | 7.310 | 80.879 | 6.252 |

Full-chain A/B is five ABBA cycles against the old non-prefetched gate.
Stage timings alternate forward/reverse measurement order over ten samples;
drop each stage's smallest/largest sample. Each sample is100 graph replays.
Individual graph times must NOT be summed into a service critical-path budget.
Sorter and first input quant are outside this chain, as in the existing oracle.
The current end-to-end baseline was not remeasured in this experiment.

## Decision

Gate remains the largest compute component after prefetch. Quant+reducer are
only about13us in isolation. Existing negative gate-quant fusion, token-major
direct-final down and multi-bucket launches remain negative evidence; do not
repeat them merely to remove a launch. Next candidate needs to change useful
gate work/weight access without increasing accumulator ownership or persistent
weight cache. Validate on real diverse-decode routing before service promotion.

A contemplated LUT-before-work early exit is also not a useful current M32
direction: down has2048 row tiles per A4 scan and64 subgroups per CTA. All832
CTAs have work once there are26 scans. Even this concentrated fixture has61
scans, so a CTA-empty fast exit cannot save its LUT initialization.

Reproduce with:

```sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/home/pc/Code/sglang/python \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --breakdown
```

Adjacent JSON retains all timing samples. Temporary probe allocations were
released at process exit; no additional service workspace or KV reduction.
