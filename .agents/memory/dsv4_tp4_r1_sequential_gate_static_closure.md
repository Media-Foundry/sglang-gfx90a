# TP4 M32 A4/R1 sequential gate-then-up static closure (2026-08-30)

The oracle keeps the production packed-FP4 LDS LUT, A4 assignments, eight
waves, original group/K order, SDOT arithmetic and wave64 shuffle tree.  Each
wave computes one gate row, stores its four reduced FP32 gate values in
wave-private LDS, then makes a second projection pass for up and writes the
original bounded-SwiGLU BF16 result.

The standalone source is
`python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_gate_up_r1_sequential_oracle.cuh`;
it is not connected to production.

The gfx90a descriptor for `E256/M32/T6/I512/K4096/A4/R1/W8/B2080/LUT2`
compiled successfully but reports:

- **85 VGPR**;
- 48 SGPR;
- 1152 bytes LDS;
- zero private segment and zero VGPR/SGPR spill;
- 512-thread maximum workgroup.

This is well above the predeclared `<=64 VGPR` continuation gate.  Reducing
rows from two to one does not shorten the live packed-weight decode/address
state enough; the sequential second projection pass also retains more state
than the hoped-for four accumulators alone.  Per instruction, no GPU ABBA,
mutation replay, service wiring or geometry sweep was run.  Stop this
representation at the static descriptor.

