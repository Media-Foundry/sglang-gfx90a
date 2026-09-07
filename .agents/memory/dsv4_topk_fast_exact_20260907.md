# Exact HIP Top-K acceleration candidate, 2026-09-07

Status: default-off mode 3; production remains mode 2. No E2E speedup or
full-model precision claim from this experiment. Running TP8 service unchanged.

## Contract and implementation

Same FP32 score bits and 64-bit ordered key as mode 2: descending score,
lower logical ID wins ties, selected IDs emitted in descending logical order.
Signed zero normalized; NaN policy unchanged. Physical page mapping performed
after selection. No epsilon, approximation, or changes to model weights.

Candidate uses four wave-private counting histograms, integer parallel bucket
prefix scan, and contiguous descending-ID intervals per thread for emission.
Atomics count only; never determine tie membership. Replaces serial bucket
scan and repeated block publication barriers. Source:
`python/sglang/kernels/jit/csrc/deepseek_v4/topk_deterministic_hip.cuh`.

Mode 3 is selected by `SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER=3`.
Static shape guard: fast only for batch <=32 OR score width <=576; otherwise
mode 2 kernel. No D2H occupancy/length sync. Default remains 2.

## GPU component ABBA

GPU0; amd-smi PID check before tests; resident TP8 service idle. Fixed random
scores, same tensors, 50 sequential nodes per graph, 20 replays per arm.
Times are microseconds per Top-K call, NOT E2E. Both A arms mode 2, B mode 3.

| Batch | Logical candidates | Reference us | Raw candidate us |
|---:|---:|---:|---:|
| 1 | 576 | 35.582 | 8.822 |
| 1 | 4096 | 50.454 | 26.406 |
| 1 | 16384 | 125.680 | 108.507 |
| 1 | 65536 | 426.955 | 376.472 |
| 32 | 576 | 39.071 | 9.087 |
| 32 | 4096 | 50.854 | 27.362 |
| 32 | 16384 | 152.169 | 128.953 |
| 32 | 65536 | 604.884 | 559.369 |
| 2304 | 576 | 119.750 | 45.778 |
| 2304 | 4096 | 257.805 | 555.714 |

Important negative result: raw fast kernel at M2304/L4096 loses over 2x.
Likely poor coalescing of per-thread interval rereads; not hardware-counter
confirmed. Do NOT deploy unguarded or assume all prefill improves. Shape guard
retains reference kernel for this case. Full ABBA raw/guarded data in adjacent
JSON. Uniform full-length synthetic prefill is not real ragged prefill.

## Correctness after final selector change

`scripts/rocm/check_dsv4_deterministic_topk.py --mode 3 --replays 1000 --mutations 1000`
passed:

- independent CPU stable-sort reference for random, all-tied, quantized scores;
- ragged/empty/short lengths, score/page strides, logical and physical outputs;
- real layer8/row2144 cutoff-tie regression fixture;
- 32K/64K all-tied rows; each small suite 1000 graph replays with output poison;
- mode2/mode3 graphs, 1000 mutations of scores, live lengths, and page mapping;
  includes signed zero, infinity, NaN, many exact ties; outputs bitwise equal;
- M2304/L576 fast and M2304/L4096 fallback: CPU reference + 10 graph replays.

## Next acceptance gate

Fresh TP4 mode2/mode3 E2E ABBA with real diverse prompts, fixed teacher-forced
token/logprob comparisons, long prefill and C16; then TP8. Do not promote from
component results alone. Short prompts <=index_topk skip Top-K entirely, so
short C1 decode throughput cannot substantiate this improvement.
