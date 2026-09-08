# Live TP8 prefetched gate code-object audit

Baseline after recorder removal: PID3243459, FranceC32 32/32 exact;
restore job `/tmp/dsv4_tp8_occupancy_restore_20260908_state.json` validated.
No production kernel, selector or capacity changed during this audit.

Read the mapped library from the live TP0 scheduler rather than selecting a
cache file by modification time. Loaded module stem:
`sgl_kernel_jit_gfx90a_fp4_expert_gate_row_prefetch_256_32_6_256_4096_4_2_8_832_2`,
build `261154e0a80a1f39`, dependencies `3dee9d5c83c361a0` under
`/home/pc/.cache/sglang/jit/gfx90a/`.

System `roc-obj-ls` Python wrapper raises ImportError; did not alter packages.
Used `/opt/rocm/core-7.14/llvm/bin/llvm-objdump --offloading <so>` to extract
the gfx90a code object, then llvm-readobj --notes and llvm-objdump -d.
Extraction creates adjacent derived code-object files, not changes to the .so.

Actual gfx90a:sramecc+:xnack- kernel metadata:

- VGPR95, SGPR51, wave64;
- LDS1024 bytes, private segment0;
- VGPR/SGPR spill counts0.

Static disassembly has1371 instructions,128 `v_dot4c_i32_i8_e32`,
21 `global_load_dwordx4`,9 `global_load_dword`,2 `global_load_ushort`, and no
global byte scale loads. Counts are static instructions, not dynamic execution
counts, occupancy counters or timings.

The original compiler already combines the four gate/up scale bytes into two
ushort loads. Thus the rejected paired-scale candidate did not remove a
missing vectorization opportunity. Do not attribute its slowdown to a proven
register change: only the baseline object was audited here.

The128 static SDOT instructions correspond to one A4/R2 gate+up K32-group
body (4*2*2*8), consistent with retaining the two-iteration lane-strided K
loop. No evidence supports adding a no-unroll switch to fix duplicated K
bodies. This is a diagnostic stop, not a new speed checkpoint.
