# Comb-only continuation oracle (component tested; not production-wired)

Do not run while a TP8 service owns the GPUs. This is not wired into
production. `screen.json` and `full.json` now record exact component results:
large M is5.7–5.9% lower complete-boundary latency than full20, small M loses.
See `.agents/memory/dsv4_prefill_mhc_refine20_20260915.md` for scope and limits.

The current split-K tail does initial Sinkhorn normalization followed by
`ITERS-1` row/column normalization pairs, writes FP32 comb, then computes
weighted RMSNorm using pre (which is independent of iteration count).
Thus a candidate can execute the existing8-iteration tail and continue only
comb for12 more pairs. It must NOT repeat the exponential/initialization.

Motivation: restoring20 currently adds about0.88ms per M32768 batch1 full
boundary in the config20 component test. A tiny one-wave comb continuation
might avoid keeping the large8-wave RMSNorm CTA busy for those extra pairs.
This is a hypothesis, not a gain estimate or proof that launch splitting wins.

The in-progress config20 service trial has a useful magnitude check: its
six candidate timing waves average leg medians5480.609538 versus initial
control5542.583721 input tok/s, about1.07s extra per524286-token wave.
Assuming16 large forwards,43 layers and two affected boundaries per layer,
the isolated0.881747ms increment would total1.21s. This is consistent in
scale but is not a service critical-path measurement or a completed ABBA.
The candidate aims to recover this small correctness cost, not to claim a
large throughput increase over the old8-iteration baseline.

The candidate adds one launch and an in-place read/write of16 FP32 values
per token (2MiB each direction at M32768), with no additional tensor workspace.
It preserves the existing pre-mix, Fn precision, post and normalized output.
Different compiler reduction layouts can still break bit equality: the oracle
requires all four full-boundary outputs exact, not merely mathematically
equivalent or below a tolerance. Stop on mismatch and retain the failed file.

After all service arms stop and GPU ownership is clear:

```bash
HIP_VISIBLE_DEVICES=4 SGLANG_USE_AITER=1 \
PYTHONPATH=python:python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:python/sglang/kernels/aot/python \
/home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_prefill_mhc_refine20_20260915/oracle.py \
  --sizes 1 128 8192 --mutations 10 \
  --output .agents/experiments/dsv4_prefill_mhc_refine20_20260915/screen.json
```

The first screen is allocating eager full-boundary timing only. Row permutation
checks are now included for every shape. If promising, run ragged M32767 and
M32768 and use `--graph-replays 1000` before any service integration. This flag
also tests a replay after changing inputs to detect stale graph output; zero
replays are recorded as untested, not as a graph correctness pass. The extended
run has passed25 mutations, row permutation,1000 graph replays and changed-input
replay per shape at M128/8192/32767/32768.
Captured residual/Fn are local prerequisites; post inputs
are synthetic. This is not a fresh complete model trace or an accuracy oracle.
