# DSpark M32 CKTile split-K key fix and rejection (2026-09-01)

## Scope

- Original DeepSeek-V4-Flash safetensors, TP4/EP1/no-A2A.
- Physical GCDs 4--7.
- Gamma-three DSpark, BS32, 32 distinct real code/chat requests.
- Candidate disabled the custom packed-SDOT direct MoE and requested AIter
  CKTile A16W4 with `SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT=2`.

## Tune-key bug

The first screen did not exercise split-K2 even though `/proc/<pid>/environ`
contained the requested value. AIter printed `ksplit = 0` and used its untuned
default. SGLang's injected tune key encoded:

```text
dtype, q_dtype_a, q_dtype_w = BF16, FP4, FP4
```

For `ActivationType.Dsv4Silu` on pre-gfx950, AIter deliberately keeps the
activation in BF16 and queries:

```text
dtype, q_dtype_a, q_dtype_w = BF16, BF16, FP4
```

The mismatch silently made every gfx90a injected tune row unreachable. The
key builder now mirrors AIter's dtype rule and has a CPU unit test. The full
`test_aiter_runner.py` file passes 4/4.

## Real M32 result

After restarting from the patched source, graph capture logged an explicit
two-stage row with empty kernel names for the exact M32 key, proving CKTile
selection rather than fallback. A finite 32x256 real-request screen produced:

```text
resident BS32:       714.76 tok/s
mean accepted:         2.883
France semantic:       false
length/finish gates:   pass
```

The pre-fix fallback screen was `667.48 tok/s`, acceptance `2.803`, also
France-false. Both are far below the accepted packed-SDOT DSpark checkpoint
near 1.5k. The candidate misses the required `<=320 us` full-routed budget by
a wide service margin; do not spend another service launch on split-K4.

Artifacts:

- `/tmp/dsv4_2k_cktile_m32_ksplit2_screen.json`
- `/tmp/dsv4_2k_cktile_m32_ksplit2_keyfix_screen.json`

## Decision

Keep the tune-key correctness fix so `SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT`
does what it claims. Keep the production TP4/BS32 profile on the exact custom
packed-SDOT direct MoE. The key fix is dormant on that direct-return path and
does not change native AR or the accepted DSpark default.
