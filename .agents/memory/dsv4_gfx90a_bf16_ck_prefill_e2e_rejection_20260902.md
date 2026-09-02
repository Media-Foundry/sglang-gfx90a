# DSV4 gfx90a large-M BF16 CK prefill: correctness repaired, E2E rejected

Date: 2026-09-02

## Scope

This experiment tested the default-off
`SGLANG_DSV4_GFX90A_BF16_CK_PREFILL` selector for routed MoE at
`8192 <= M <= 36864`.  It expands the original FP4 expert weights to BF16 and
uses generic CK two-stage MoE.  Decode/C1 do not enter this selector.

## Correctness root causes and repairs

Two independent failures were found:

1. Generic CK `DeviceMoeGemm` expects its BF16 B tensor in the 16x16
   preshuffled layout.  Passing row-major expanded weights made only a small
   subset of channels meaningful.  A tiled gfx90a FP4-to-preshuffled-BF16
   dequant kernel is exact against `aiter.ops.shuffle.shuffle_weight(...,
   layout=(16, 16))`.
2. CK's BF16 `AtomicAdd` implementation is empty on gfx90a.  MI200 has no
   native BF16 global atomic add.  An FP32 stage-2 accumulation workspace plus
   BF16 copy restores nonzero, numerically close output.  A packed 2xBF16 CAS
   prototype compiled but still produced zero output and is not usable.

With preshuffled weights, bounded DSV4 SwiGLU, and FP32 stage-2 accumulation:

- stage-1 max abs error versus Torch: 0.5; mean abs about 0.000993;
- routed-output cosine: 0.9999959--0.9999961;
- routed-output max abs error: 256--512 with reference absmax 76k--104k;
- no NaN, empty output, or obvious numerical collapse.

## End-to-end semantic result

Configuration: TP4/EP1/no-A2A on physical GCDs 4--7, original checkpoint,
83,968-token pool, 32 distinct code-review prompts (73,724 prompt tokens).

- France oracle: `The capital of France is Paris.`
- C32 semantic run: 32/32 answers were coherent and task-related;
  completion unique-token ratios were 0.734--0.953.
- Cross-round greedy output was not bitwise stable: first token matched for
  16/32 requests in a two-round comparison, and full 64-token completion
  matched for 2/32.  This is consistent with small floating-point differences
  being amplified by greedy autoregression, but it must be reported as a
  non-bitwise path.

## End-to-end performance result

Five C32 rounds with eight generated tokens gave aggregate input throughput:

```text
1761.6 (cold), 2322.8, 2375.0, 2344.5, 2385.2 tok/s
warm median: 2344.5 tok/s
warm trimmed mean: 2347.4 tok/s
```

This is slower than the production SDOT/FP4 path (about 2.75k tok/s) and the
original AIter FP4 comparison (about 3.11k tok/s).  The large-M standalone CK
advantage does not survive FP4-to-BF16 expansion, FP32 stage-2 workspace
zero/write/copy traffic, and the real chunked scheduler execution.

## Decision

Keep the selector default-off and do not promote this path.  The numerical
failure was repaired sufficiently for semantic evaluation, but the E2E result
is a performance regression.  Future CK work should consume packed FP4
directly or avoid the FP32 atomic workspace; a BF16-expanded full-weight path
is not a production candidate.

Artifacts:

- `/tmp/dsv4_bf16_ck_c32_semantic_round1.json`
- `/tmp/dsv4_bf16_ck_c32_semantic.json`
- `/tmp/dsv4_bf16_ck_c32_perf.json`
