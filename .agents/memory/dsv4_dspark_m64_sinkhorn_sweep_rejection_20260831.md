# DSV4 DSpark M64 native-Sinkhorn iteration sweep rejection

Date: 2026-08-31

## Scope

- Original DeepSeek-V4-Flash weights; proposed selector was restricted to
  `gfx90a + TARGET_VERIFY + M64` and was never committed or enabled for AR.
- Single-GPU oracle used physical GCD 4 after `amd-smi` reported no process.
- Native wave64 Sinkhorn kernels were captured independently at 4, 8, 12 and
  20 iterations over `[64,1,24]` FP32 mixes.

## Microbenchmark

Nine-round symmetric-order graph timing (18 samples/arm) produced:

| iterations | trimmed time | comb max abs vs 20 |
|---:|---:|---:|
| 4 | 6.47999 us | 5.96e-8 on the timing sample |
| 8 | 7.48075 us | 2.98e-8 on the timing sample |
| 12 | 8.45255 us | 0 on the timing sample |
| 20 | 10.41766 us | reference |

The apparent 12-iteration exactness did not generalize: the first of 100
independent random `mixes/scale/base` mutations differed from 20 iterations,
with comb max-abs `5.364418e-7` and 356 differing FP32 elements.  The isolated
12-vs-20 saving is only `1.965 us` per Sinkhorn call.  Even two calls per layer
over 43 layers cap the gross target-step saving near 0.17 ms if the calls are
fully serialized (and below 0.5% of the measured M64 target step).

## Contaminated service attempt

A target-only implementation was temporarily applied and immediately reverted.
The 20-iteration control service passed France and all 32 distinct coding
requests completed 256 tokens.  Its resident throughput declined
`719.46, 724.36, 696.67, 674.40, 646.47 tok/s` while acceptance stayed near
`1.60--1.64` and host steps were `61--68 ms`.

This run is invalid for performance comparison.  PID 772080
(`/home/pc/anaconda3/envs/BIO/bin/python -m
pls.training.train_editflow_gb1_confirmatory`) entered physical GCD 4 at
19:32, retained about 934 MiB there, and overlapped the benchmark.  The process
was not owned by this experiment and was not terminated.  Our candidate
service was stopped without running a performance arm.

## Decision

Reject the 12-iteration target-verify selector.  It is non-exact under varied
inputs and its sub-0.5% gross bound is too small to justify a model-level
correctness risk.  No production or AR code remains from the experiment.
Raw oracle timing is `/tmp/dsv4_dspark_m64_sinkhorn_sweep.json`; contaminated
control output is `/tmp/dsv4_sinkhorn20_a_code32.json`.
