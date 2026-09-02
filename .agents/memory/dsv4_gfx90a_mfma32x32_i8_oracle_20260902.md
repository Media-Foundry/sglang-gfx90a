# DSV4 gfx90a I8 MFMA 32x32 oracle (2026-09-02)

## Scope

Standalone-only CDNA2 oracle. No production model, AR, decode, or prefill
wiring was changed.

## ISA mapping and correctness

The tested instruction is `v_mfma_i32_32x32x8_i8`. Four K8 issues implement
an exact M32xN32xK32 integer tile. Its output lane mapping is:

```text
column = lane & 31
row = (lane >> 5) * 4 + (vector_index & 3) + 8 * (vector_index >> 2)
```

On physical GPU 4, 100 independently mutated random inputs matched both the
SDOT4 reference and CPU int32 matmul bitwise.

## ABBA result

Raw M32xN32xK32 tile, nine ABBA rounds with 10,000 launches per sample:

```text
MFMA: 3.1192 us
SDOT: 5.7724 us
speedup: 1.851x
```

This is a raw-tile result, not an end-to-end result. The next gate is a real
FP4 unpack + E8M0/group-scale A32/N32 expert tile. Production integration is
allowed only if the complete routed stage improves by at least 5% and the
normal correctness suite passes.
