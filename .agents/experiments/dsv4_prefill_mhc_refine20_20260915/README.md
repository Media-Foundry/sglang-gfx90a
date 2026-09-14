# Prepared comb-only continuation oracle (not GPU-tested)

Do not run while the config20 TP8 service ABBA owns the GPUs. This is not
wired into production, and no speed or bitwise claim has been established.

The current split-K tail does initial Sinkhorn normalization followed by
`ITERS-1` row/column normalization pairs, writes FP32 comb, then computes
weighted RMSNorm using pre (which is independent of iteration count).
Thus a candidate can execute the existing8-iteration tail and continue only
comb for12 more pairs. It must NOT repeat the exponential/initialization.

Motivation: restoring20 currently adds about0.88ms per M32768 batch1 full
boundary in the config20 component test. A tiny one-wave comb continuation
might avoid keeping the large8-wave RMSNorm CTA busy for those extra pairs.
This is a hypothesis, not a gain estimate or proof that launch splitting wins.

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

The first screen is allocating eager full-boundary timing only. If promising,
add ragged/row-permutation/mutation coverage and1000 graph replays before any
service integration. Captured residual/Fn are local prerequisites; post inputs
are synthetic. This is not a fresh complete model trace or an accuracy oracle.
