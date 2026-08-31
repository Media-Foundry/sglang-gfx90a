# DSpark gamma-3 M128 CK sparse-decode rejection (2026-08-31)

## Scope and guard

- Original DeepSeek-V4-Flash weights, TP4/EP1/no-A2A, physical GCDs 4--7.
- DSpark gamma 3, BS32 target verification (`M=128`).
- Prototype selector required an explicit backend-provided
  `target_verify=True` plus T128/H16/D512/BF16/TP4 and a dedicated env flag;
  native AR could not match it.  All prototype wiring was removed after the
  final E2E rejection.

## Initial standalone result

The existing CK-style MFMA split-2 kernel was generalized from max M96 to
M128.  Against the existing oracle's forced Triton split-4 geometry:

```text
context 128: Triton 125.835 us, CK  77.088 us, save  48.747 us
context 256: Triton 191.960 us, CK 119.436 us, save  72.524 us
context 512: Triton 306.952 us, CK 182.222 us, save 124.730 us
```

All three contexts passed 100 randomized Q mutations within the established
`max_abs <= 0.0078125`, `relative_l2 <= 0.005` contract and 1000 HIP Graph
replays with bitwise-stable CK output.

## Corrected production comparison

Three real heterogeneous BS32 rounds with the CK selector produced:

```text
891.940 / 856.807 / 892.973 tok/s; median 891.940
```

The adjacent Triton control was:

```text
912.428 / 839.779 / 901.625 tok/s; median 901.625
```

Acceptance-normalized medians also favored control (about 372.49 versus
368.67).  Every round passed France first-nine exact + semantic Paris and all
32 requests returned exactly 256 tokens with `finish=length`.

Layer-20 realtime markers exposed the false oracle:

```text
CK target attention core:          about 68--71 us
production Triton attention core:  about 68--79 us, warm samples about 68 us
```

At M128, production computes `T * H = 2048` head rows and the heuristic selects
the single-pass fused `kv_splits=1` Triton kernel.  The standalone oracle had
explicitly forced `kv_splits=4`, so its 40% CK advantage compared against a
non-production Triton geometry.  Once compared to fused split-1, CK split-2
has no meaningful critical-path advantage.

## Decision

- Reject M128 CK split-2 and restore the M96 wrapper cap and selectors.
- Do not use `bench_dsv4_tp4_m32_ck_sparse_decode.py` with forced split-4 to
  justify production M128 changes.  Future M128 attention oracles must compare
  against the exact production heuristic (`kv_splits=1`).
- The current production Triton attention core is already about 68 us at this
  tier; prioritize the roughly 280-us prepare, 195-us attention epilogue, and
  635--670-us MoE/FFN regions instead.

