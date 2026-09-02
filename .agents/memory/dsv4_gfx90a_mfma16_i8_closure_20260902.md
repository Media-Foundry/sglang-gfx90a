# DSV4 gfx90a native M16 I8 MFMA closure (2026-09-02)

## ISA oracle

A direct CDNA2 `v_mfma_i32_16x16x16_i8` oracle was added and compared with
the production-style `v_dot4_i32_i8` reference.

- M16N16K32, 100 random mutations: bitwise exact.
- M16N16K32: MFMA 2.4512 us, SDOT 3.0246 us, 1.234x isolated speedup.
- Raw-FP4 + E8M0 A16N16K4096, 100 random mutations: bitwise exact.
- A16N16K4096: MFMA 51.501 us, SDOT 472.411 us, 9.173x isolated speedup.

The FP4 oracle retains the production contract: doubled signed-I8 codebook,
per-group E8M0 scale, and the required 0.5 correction.

## Closure

The existing production-shape function named
`gfx90a_fp4_expert_gate_up_mfma32_kernel` already uses this exact native
M16N16K16 instruction.  The `32` in its name is the expert-block assignment
count, not the MFMA instruction dimensions.  That complete kernel was already
measured at 29.000 ms versus 13.926 ms for production SDOT at M4608.  Its
resource cost and full scheduling overhead erase the isolated instruction
gain.

Therefore this is an ISA/correctness oracle only.  Do not create another CK
wrapper around the same instruction or reconnect it to the service unless the
work decomposition changes enough to address the complete-kernel resource
problem.
