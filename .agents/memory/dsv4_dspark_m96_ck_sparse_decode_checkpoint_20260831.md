# DSV4 TP4 DSpark M96 CK sparse-decode checkpoint (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash checkpoint, TP4/EP1/no-A2A.
- Physical GCDs `HIP_VISIBLE_DEVICES=4,5,6,7`; standalone oracle on GCD 4.
- DSpark gamma two, static verify, target graph tier M96.
- 32 distinct coding prompts from
  `.agents/memory/dsv4_tp4_code_32_input_ids.json`
  (`sha256=376bf3caa2b43ebae7e20086c41e6ab028258e35734b1350630f935ec63d2ebb`).
- The new selector is default off and affects only TP4, BF16 C128 sparse decode
  with exactly 96 query rows. Gamma-one M64 remains unchanged.

## Kernel generalization and oracle

The existing CK-style gfx90a MFMA sparse-decode core already launches its grids
from `args.tokens`; only the wrapper and selector were hard-coded to M64. The
wrapper now accepts positive token counts through 96 and the production selector
has a separate `SGLANG_DSV4_GFX90A_TP4_M96_CK_SPARSE_DECODE` gate.

On physical GCD 4, with 16 TP4-local heads and D=512:

| context rows | Triton | CK/MFMA | saving | speedup |
|---:|---:|---:|---:|---:|
| 128 | 97.940 us | 66.465 us | 31.475 us | 47.36% |
| 256 | 150.643 us | 101.803 us | 48.841 us | 47.98% |
| 512 | 247.396 us | 159.827 us | 87.569 us | 54.79% |

Three context sizes passed 100 randomized Q mutations. Maximum absolute error
versus the established Triton path was 0.0078125 and maximum relative L2 was
0.00381. Contexts 128 and 512 additionally passed 1000 HIP Graph replays with
bitwise-stable CK output.

## Independent-service ABBA

Each arm used the same seed and ran three 32-request, 256-token rounds. Arm A
disabled the M96 selector; arm B enabled it. Combined medians across A1/A2 and
B1/B2 were:

| metric | A: Triton M96 | B: CK M96 | change |
|---|---:|---:|---:|
| scheduler decode | 656.315 tok/s | 693.303 tok/s | **+5.63%** |
| host speculative step | 91.140 ms | 89.104 ms | **-2.23%** |
| aggregate HTTP | 624.428 tok/s | 638.419 tok/s | +2.24% |
| common-resident HTTP | 582.244 tok/s | 580.498 tok/s | -0.30% |
| mean accepted length | 2.14430 | 2.14159 | -0.13% |

All four independent services passed the exact France first-nine-token oracle
and semantic Paris check. Every one of the 384 measured coding requests returned
256 tokens with `finish=length`. The pre-existing asynchronous path still does
not make every long completion bitwise identical across independent rounds; no
new France or request-completion failure was introduced by the selector.

## Decision

Keep the M96 selector available but default off. It is a real target-model
kernel and scheduler improvement for gamma two, but gamma two remains slower
than the accepted gamma-one BS32 checkpoint (roughly 693 versus 773 scheduler
tok/s). Revisit it when a per-request gamma/compact policy makes M96 a useful
part of the final throughput profile; do not replace gamma one globally.

Artifacts:

```text
/tmp/dsv4_m96_{a1,b1,b2,a2}_{france,code32}.json
/tmp/dsv4_dspark_seed541_realcode32.json
```
