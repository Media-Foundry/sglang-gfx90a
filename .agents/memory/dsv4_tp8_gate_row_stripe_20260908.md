# TP8 prefetched gate row-stripe scheduling: rejected

Baseline b49ecd73b1, native TP8 I256 M32/A4/R2/W8/G832, accepted row-prefetched
gate and unchanged subgroup8 down. No production selector or serving process
changed. The service retains its 1M KV pool and original checkpoint weights.

## Candidate

Traverse a short stripe of output-row tiles, then the next sorted A4 chunk,
instead of traversing all128 row tiles of each chunk consecutively. Adjacent
chunks of the same expert may reuse weight rows. No new accumulator, barrier,
atomic, workspace or weight cache. SDOT/reduction and output addresses unchanged.

The implementation is compiled only with
`SGLANG_FP4_GATE_ROW_STRIPE_ORACLE`; normal compilation retains the original
mapping. The standalone module has a distinct JIT key including stripe width.
There is no production environment switch.

## Complete chain microseconds, trimmed five-cycle ABBA

| Expert pool / active / scans | Original / stripe8 | Original / stripe32 |
| --- | ---: | ---: |
| 256 / 133 / 133 | 274.980 / 304.798 | 274.993 / 285.811 |
| 128 / 104 / 105 | 241.936 / 273.232 | 241.126 / 256.803 |
| 32 / 32 / 61 | 182.880 / 193.925 | 182.869 / 193.646 |

Stripe8 regresses6.0–12.9%; stripe32 regresses3.9–6.5%. No E2E integration.
Potential cache reuse did not offset the changed task ordering/address work.
No hardware counter evidence separates those causes; do not assert one root
cause from timings alone. This rejects these two mappings, not every possible
expert-run design.

## Correctness and limits

Each of six cases passed100/100 exact gate intermediate, down FP32 partial and
final BF16 comparisons under activation/scale mutation. Replay output remained
stable. CPU enumeration proved mapping bijection for1..192 valid A4 blocks and
stripes8/16/32/64/128; empty count never enters the task loop.

Inputs, weights and routing are synthetic, not a real decode tensor fixture.
As in the existing oracle, sorter and first input quantization are outside the
timed chain. The experiment rejects candidates at component level; no claim of
new model correctness or changed C1/C32 E2E throughput is made.

AMD-SMI ownership was checked before the run; GPU4 only. The initial execv
launcher failed on an import before GPU work; direct DS interpreter with the
repository PYTHONPATH ran both tests successfully. No dependency was installed.

```sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/home/pc/Code/sglang/python \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --gate-row-stripe 8
# Repeat with --gate-row-stripe 32.
```

Adjacent JSON preserves all samples. Keep current production task order.
