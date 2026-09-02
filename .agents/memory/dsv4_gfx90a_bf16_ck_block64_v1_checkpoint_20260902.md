# DSV4 gfx90a BF16-CK block-M64 V1 checkpoint (2026-09-02)

## Scope

- DeepSeek-V4-Flash original weights, TP4/EP1, no A2A, GCD4-7.
- Default-off large-prefill profile only:
  `SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1`.
- The latency/default AR profile is unchanged.

## Root cause and implementation

At the real C32 admission shape (`M=27648` routed rows), CK stage 1 was the
dominant BF16-CK cost.  AIter's generic gfx90a block-M64 heuristic selected a
generated instance which returned all zeros.  An explicit valid instance was
found instead:

```text
moe_ck2stages_gemm1_256x64x64x128_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight0_dsv4silu_B16_B16_B16
```

The production profile now enables it through the fixed boolean
`SGLANG_DSV4_GFX90A_BF16_CK_BLOCK64_V1=1`; arbitrary kernel-name and block-M
environment variables remain diagnostic overrides.  Stage 2 retains the
correct FP32 workspace because gfx90a has no native BF16 atomic add.

## Standalone results (physical GPU 4)

With the formal boolean selector, M27648 and 256 dequant blocks:

| Routing | Dequant | CK core | Total |
|---|---:|---:|---:|
| balanced | 4.933 ms | 27.026 ms | 31.889 ms |
| skewed | 4.929 ms | 23.827 ms | 28.647 ms |

The earlier strict A/B against block-M32 measured:

- balanced: 55.087 -> 31.480 ms (42.9% faster), bitwise exact;
- skewed: 40.889 -> 28.282 ms (30.8% faster), cosine 1.0 and mean absolute
  difference `4.37e-5` from the existing FP32 atomic reduction ordering.

## End-to-end results

On 32 fixed heterogeneous real code requests with the 12+12+8 admission
profile, five rounds produced:

```text
5080.77 (cold), 5864.54, 6075.08, 5744.65, 5450.60 input tok/s
```

The five-round median is 5744.65 input tok/s.  This is 7.2% above the previous
correct req12 block-M32 median of 5359.4 tok/s, and 18% above the older req16
baseline of 4865.5 tok/s.

C1 prefill is not selected by the large-M path and remains above the target:
2575.1 / 2590.2 / 2590.1 input tok/s for a 2304-token prompt.

Native AR correctness/decode regression:

- official token-ID France oracle: first nine IDs exact in both rounds;
- semantic answer contains Paris;
- completion SHA256 identical across rounds;
- scheduler-native decode: 58.06 tok/s in the measured round.

## Remaining work

The C32 target is still 10k input tok/s.  Profile the accepted block-M64 path
again: stage 2 and the fixed dequant pass are now the next local costs.  Any
further CK instance substitution must be evaluated on the complete routed
stage and followed by the same France/C1/decode regressions.
