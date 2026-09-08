# Accepted-prefetch gate CTA-only screen

Baseline 028679f7eb, native TP8 service PID 3223021 retained unchanged with
1,048,576-token pool. AMD-SMI audit found no external GPU owners. Standalone
probe ran on physical GPU4 and exited normally; no service weight cache added.

Candidate changes only gate CTA count 832 to 624, retaining accepted
prefetched A4/R2/W8/I256 gate, INT8 quant, ordinary down and fixed reducer.
No row-stripe or down-prefetch change. Synthetic weights/inputs and three
synthetic routing distributions; not a captured real-decode routing fixture.

Full gate→quant→down→reducer chain, five ABBA cycles, 100 graph replays per
sample, discard min/max:

| Active experts / scans | 832 CTA us | 624 CTA us |
|---|---:|---:|
|133 / 133|275.6792|276.5226|
|104 / 105|242.2396|241.2582|
|32 / 61|182.9830|181.6884|

All three cases pass 100 mutation comparisons each: BF16 intermediate/final
and FP32 partial exact, max_abs=0, replay stable. Baseline first-case timing
has a 334.64 us outlier preserved in JSON, removed only by stated trimming.
No robust full-chain benefit (roughly -0.31% / +0.41% / +0.71% throughput).
Do not promote or claim E2E speedup. Production retains 832 CTA. No further
CTA grid sweep is justified by this small mixed result alone.

Added standalone `--gate-blocks` option with incompatible experiment modes
rejected. Reproduce:

```sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/home/pc/Code/sglang/python \
 /home/pc/anaconda3/envs/DS/bin/python \
 scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --gate-blocks 624
```
