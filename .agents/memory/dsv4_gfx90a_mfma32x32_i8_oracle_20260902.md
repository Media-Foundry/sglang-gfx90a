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

## Real FP4 tile and production-shape closure

The raw-checkpoint-layout `A32 x N32 x K4096` tile was then extended with real
FP4 unpacking, E8M0 weight scales, group-32 activation scales, and FP32
cross-group accumulation. It matched an SDOT4 implementation bitwise across
100 independently mutated inputs. The isolated tile measured `95.425 us`
versus `1887.789 us` for the intentionally simple scalar-output SDOT oracle.
That large ratio was not treated as a production claim.

A complete M4608 gate/up candidate consumed the production A64 sorter as two
A32 halves and was compared directly with the established production kernel:

| blocks | production | MFMA32x32 candidate | regression |
|---:|---:|---:|---:|
| 416 | 13.926 ms | 29.000 ms | 52.0% slower |
| 624 | 13.833 ms | 30.494 ms | 54.6% slower |
| 832 | 13.800 ms | 34.290 ms | 59.8% slower |
| 1248 | 13.945 ms | 40.349 ms | 65.4% slower |

The candidate used 126 VGPR, 42 SGPR, no scratch, and one wave per CTA. It held
both gate/up 16-element integer accumulators plus both 16-element FP32 scaled
accumulators, while B loads remained separated by the 2048-byte output-row
stride. The production-shape loss therefore closes this schedule despite the
fast isolated instruction tile. Its BF16 output differed at 10,518 positions
with maximum absolute error 0.5 because the cross-group FP32 reduction order
was changed; it was never connected to the model.

## ISA and CK audit

The MI200 ISA contract was independently checked: four
`v_mfma_i32_32x32x8_i8` issues correctly cover K32, and the implemented A/B/C
lane mapping, control operands, scales, and FP4 codebook compensation are
valid. LLVM builtin scheduling owns the documented 16-pass accumulator hazard;
manual `s_nop` insertion is not appropriate.

Do **not** replace this with CK's existing named
`WarpGemmMfmaI8I8I32M32N32K32...` on gfx90a. That older CK branch converts I8
elements to FP32 and loops over `mfma_f32_32x32x2f32`; native I8 is used only
by later architectures. A proper CK implementation would need a new gfx90a
32x32x8-I8 attribute and cooperative B movement. Merely wrapping the same
builtin with CK cannot repair the measured schedule regression. The complete
gate result fails the 5% continuation gate, so no production or service test
is warranted.
