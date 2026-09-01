# DeepSeek V4 Flash TP4/BS32 DSpark 1.5k checkpoint (2026-09-01)

## Scope

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- Model: original `/home/pc/models/modelscope` safetensors
- Parallelism: TP4, EP1, no A2A
- DSpark: gamma/block size 3, target graph tiers 1--32
- KV pool: 49,152 tokens, `mem_fraction_static=0.96`
- Workload: 32 distinct code/chat prompts, 1,024 returned tokens/request,
  `stream_interval=1`; no repeated-prompt synthetic batch
- Metric: resident full-BS32 output tokens divided by the resident wall window

## Accepted composition

Two individually small, exact gfx90a changes become material when composed on
the DSpark draft/verify schedule:

```bash
AITER_GFX90A_AR_1M_BLOCKS=12
SGLANG_DSV4_GFX90A_M32_GATE_ROW_PREFETCH=1
```

The first reduces the 1-MiB AIter custom-all-reduce launch from the generic
80-CTA geometry to 12 CTAs without changing its owner or reduction order.  The
second issues both same-group R2 gate/up rows before consuming row zero in the
exact M32 routed kernel.  Both defaults are deliberately installed only under
`start-dspark` plus `SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1`; native AR does not
inherit them.

## ABBA rollback evidence

The same service profile and input manifest were used for the candidate and
the rollback (`AR_1M_BLOCKS=80`, row prefetch disabled):

```text
candidate: 1531.62 / 1547.83 / 1544.34 tok/s
control:   1439.50 / 1472.90 / 1378.69 tok/s

candidate mean / median: 1541.26 / 1544.34 tok/s
control mean / median:   1430.37 / 1439.50 tok/s
mean / median uplift:    +7.75% / +7.28%
candidate France:        3/3 semantic Paris
```

Artifacts:

- `/tmp/dsv4_gamma3_ar12_rowprefetch_49k_1024_r3.json`
- `/tmp/dsv4_gamma3_control80_noprefetch_49k_1024_r3.json`

The older isolated row-prefetch A/B was correctly rejected at only +0.58%, and
the isolated 12-CTA experiment was also too small/noisy.  This checkpoint does
not overturn those single-variable results; it records a reproducible positive
interaction in the combined schedule.

## No-manual-override verification

After promoting the two defaults, the service was restarted with both
variables removed from the caller environment.  `/proc/<pid>/environ`
confirmed that the launcher supplied `12` and `1`.  Three more real-request
rounds produced:

```text
1506.03 / 1491.89 / 1514.87 tok/s
mean / median: 1504.26 / 1506.03 tok/s
France:        3/3 semantic Paris
length:        32/32 x 1024 in every round
finish:        length for every request
```

Artifact: `/tmp/dsv4_gamma3_default_promoted_49k_1024_r3.json`.

Across the six candidate/default rounds, the center is 1522.76 tok/s mean and
1523.24 tok/s median; five of six individual rounds exceed 1500, and the lone
1491.89 round is 0.54% below the threshold.  Because DSpark accepted length is
content-dependent, report the multi-round center and the individual samples;
do not present only the best late-bin value.

## Correctness interpretation

All accepted rounds used the original weights, returned the requested token
count, ended with `finish=length`, and passed the France/Paris semantic oracle.
Sampling makes cross-round token hashes intentionally non-identical, so
`cross_round_all_exact=false` is not a failure under this harness.  These
switches preserve the existing arithmetic order and are not approximate
Top-K/expert-dropping optimizations.
