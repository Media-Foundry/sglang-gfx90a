# TP4 M64 W4A4 two-digit SDOT8 oracle rejection (2026-08-30)

## Scope

This was a strict, new-files-only compile/math oracle for the real TP4 M64
routed gate/up shape.  It did not alter a production selector, checkpoint
weights, or permanent weight storage.  The proposed activation format was
uniform signed Q4 per 32 values (`absmax / 7`); therefore this was an
approximate-activation candidate even though the stored E2M1 weight values
were represented exactly.

The predeclared early-stop conditions were:

- gfx90a must emit native signed `dot8`;
- the E2M1 two-digit identity must pass exhaustively;
- online correction generation must remain near six bit operations per eight
  weights;
- the full gate kernel must use at most 96 VGPR and no scratch;
- only after those gates would the real pass20/layer34 ABBA continue, requiring
  at most 382 us versus the roughly 425 us Q8/LDS/SDOT4 gate baseline.

## Exact integer decomposition

With the usual integer E2M1 codebook

```text
{0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}
```

every weight integer is represented exactly as

```text
value = main_i4 + 16 * correction_i4
```

Only three values need a nonzero correction:

```text
+8  = -8 + 16 * (+1)
+12 = -4 + 16 * (+1)
-12 = +4 + 16 * (-1)
```

All other values use correction zero.  A GPU exhaustive oracle covered all
16 E2M1 nibbles times all 16 signed-I4 activation values (`-8..7`), 256 cases,
and passed exactly.

ROCm clang 23 compiled `__builtin_amdgcn_sdot8` for gfx90a into native
`v_dot8c_i32_i4_e32`; thus compiler/ISA support itself is not the blocker.

## Static rejection evidence

The direct online eight-nibble encoder was disassembled before any large GPU
benchmark.  Its kernel body contained:

```text
104 vector bit/arithmetic packing operations
173 total vector instructions
89 scalar control/bit operations
VGPR 19, AGPR 0, scratch 0 (encoder-only probe)
```

This is far beyond the approximately six bit operations per eight weights
allowed by the early-stop gate.  A second compile used a 256-entry, 1 KiB LDS
pair LUT plus four `v_perm` operations to avoid the branchy direct code.  That
full M64 A4/R2/W8 gate specialization compiled, but metadata reported:

```text
gate:     VGPR 121, SGPR 54, LDS 1024 B, scratch 0
quantizer: VGPR 10, SGPR 16, LDS 0, scratch 0
```

The gate therefore also fails the explicit `VGPR <= 96` continuation gate.
The two-digit representation needs two SDOT8 instructions per eight weights;
for 32 weights this is eight dot instructions, exactly the same dot-instruction
count as the existing eight SDOT4 operations.  Its only prospective gains are
half-sized activation reads and possibly cheaper decode, while the second
digit, LUT assembly, and much higher register pressure add work.  The static
evidence is consequently not promising enough to justify a real-weight M64
ABBA.

## Decision

Rejected before the large benchmark, as required by the predeclared gate.
No down kernel, production selector, service benchmark, or permanent repacked
weight buffer was created.  The temporary implementation and compile probe
were removed; this record is the only retained repository artifact.

