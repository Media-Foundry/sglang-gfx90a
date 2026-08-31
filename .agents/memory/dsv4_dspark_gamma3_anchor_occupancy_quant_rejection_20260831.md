# DSpark gamma-3 anchor occupancy and quant-only rejection (2026-08-31)

## Scope

- DeepSeek-V4-Flash original weights, TP4/EP1/no-A2A, physical GCDs 4--7.
- DSpark gamma 3, BS32 target verification (`M=128`, four rows/request).
- Workload: 32 distinct real code-generation prompts, not repeated prompts.
- Production optimization remained strict speculative-only: the existing model
  guard requires `TARGET_VERIFY`, BS32, width 4 and `[128,4096]`.

## Post-router occupancy measurement

The normal `stat` recorder only retained full-M128 counts before the model-side
anchor mask.  A diagnostic `per_token` service recorded 40 complete M128 target
forwards (1,720 layer samples), then the analysis selected rows `0::4` offline.
Each layer therefore contains exactly 192 valid anchor assignments.

All-layer means (p50 in parentheses):

```text
active experts        107.18 (108)
effective experts      67.77 (70.62)
run_len == 1 experts   63.88 (64)
run_len == 2 experts   24.92 (25)
run_len 3--4 experts   13.92 (14)
run_len > 4 experts     4.45 (4)
max occupancy          10.22 (8)

A2 weight blocks       134.24 (135)
A4 weight blocks       113.17 (114)
A8 weight blocks       108.25 (109)
```

The first three hash-router layers are even more diffuse: mean 125.05 active
experts and 80.72 singleton experts.  Learned layers average 105.84 active
experts and 62.62 singletons.  This proves that real heterogeneous BS32 traffic
does not contain long expert runs.  A2 increases real weight scans by about
18.6%; A8 removes only about 4.4% relative to A4 while greatly increasing
padding and accumulator pressure.  Keep A4.

Raw summary: `/tmp/dsv4_gamma3_anchor_occupancy.json` (ephemeral host path).

## Anchor-only activation quant experiment

A strict target-only prototype retained the physical M128 routed layout and
existing A4 sorter/expert kernels, but group-32-quantized only rows
`0,4,...,124`.  It did not compact expert compute to M32.

Micro correctness and timing on physical GCD4:

```text
100 random mutations: anchor INT8 and FP32 scales bitwise equal to Triton
1000 HIP Graph replays: bitwise stable
100 graph-input mutations: bitwise equal to Triton

Triton full-M128 quant      ~38.0 us
HIP anchor-only quant        ~9.33 us
isolated saving             ~28.7 us/layer
```

Both 256-CTA and coexistence-oriented 48-persistent-CTA versions had the same
~9.33-us standalone time.  The first service attempts appeared to regress to
~488 tok/s, but this was an invalid configuration: the profile variable was
misspelled (`GFX90A_TP4_BS32_PROFILE` instead of
`SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE`), so `--enable-single-batch-overlap` was
absent.  Explicitly restoring SBO recovered the control to 890.28 tok/s.

Corrected sparse-tier comparison (`cuda graph tiers 1,32`, 32K pool, SBO on):

```text
control rounds:   912.43 / 839.78 / 901.62 tok/s; median 901.62
candidate rounds: 851.43 / 932.72 / 895.65 tok/s; median 895.65
raw median delta: -0.66%

acceptance-normalized median:
control   ~372.49
candidate ~367.3
```

Every accepted round passed France first-nine exact + semantic Paris and all
32 requests returned exactly 256 tokens with `finish=length`.

## Decision

- Reject and remove the anchor-only quant kernel/carrier/env wiring.  The
  isolated 28.7-us saving does not shorten the actual dual-stream critical
  path.
- Retain A4 and the existing gamma-3 anchor-only routed checkpoint.
- For every future launch, set and verify the exact profile variable and check
  the resulting process contains `--enable-single-batch-overlap`; a missing
  SBO flag halves throughput and can masquerade as a kernel regression.
- Native AR was never eligible for the candidate selector and no AR default was
  changed.

