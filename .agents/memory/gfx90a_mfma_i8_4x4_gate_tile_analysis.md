# CDNA2 `v_mfma_i32_4x4x4i8` audit and real gate-tile oracle

Date: 2026-08-27

## ISA and compiler audit

- The local MI200/CDNA2 ISA manual, sections 12.10 and 13.3.6.1, lists
  `V_MFMA_I32_4X4X4I8` opcode 82 as sixteen independent 4x4x4 I8 blocks,
  two passes, one architectural VGPR for each input, and four accumulator
  VGPRs for C/D.
- ROCm clang declares
  `__builtin_amdgcn_mfma_i32_4x4x4i8` with signature
  `V4i(i32, i32, V4i, imm cbsz, imm abid, imm blgp)`. LLVM marks all three
  control operands immediate and the intrinsic convergent/no-memory.
- `cbsz` is legal from 0 through 4. LLVM's AMDGPU dialect further documents it
  as log2 of the source-A lane chunks and `abid` as the selected chunk; thus
  the default non-broadcast form is `cbsz=0, abid=0`.
- `blgp` controls source-B lane permutation: 0 none, 1/2 broadcast first/second
  32 lanes, 3 rotate 16 right, and 4--7 broadcast the first through fourth
  16-lane group. CK and rocWMMA use `0/0/0` by default, which is also the form
  used by the oracle.
- CK exposes both logical M4xN64 and M64xN4 transforms through the same builtin,
  consistent with the instruction's sixteen independent blocks.

## Measured lane and accumulator layout

A raw 64-lane probe with distinct matrices established the layout instead of
assuming it:

- lanes are partitioned into sixteen contiguous four-lane blocks;
- within a block, `lane % 4` is the B/output column;
- the returned `V4i` components are the four A/output rows.

For routed A4 this means one wave can compute an A4xN64 tile without M/N
padding: every four-lane block owns four output rows, each lane supplies one
weight row, and all blocks consume the same four assignments. For K32, eight
successive K4 MFMA instructions produced the same 4x4 int32 tile as eight
`v_dot4c_i32_i8` operations and CPU int32 matmul, elementwise exact. Applying
dyadic assignment and row scales was also FP32 exact.

Disassembly confirmed the expected instructions. The minimal correctness
kernel contains eight `v_mfma_i32_4x4x4i8` and eight `v_dot4c_i32_i8`; operands
and accumulator are the expected scalar i32/scalar i32/V4i layout.

## Real A4xN64xK4096 tile

The real-shape oracle keeps the production representation and math:

- A4 INT8 activation and 128 FP32 group scales;
- packed FP4 weights and the same 1 KiB CTA-local pair LUT decoder;
- E8M0 weight scales in `[N64,128]` layout;
- eight K4 MFMA instructions per group32, then
  `int32 * x_scale * weight_scale * 0.5`;
- 128 group contributions accumulated in FP32;
- K split 1/2/4/8 waves, at most 8 KiB LDS partials and one barrier;
- timed kernels never write the 128x4x64 integer correctness tensor.

The sdot reference retains the current 512-thread, 16-lane subgroup mapping,
group-strided accumulation and shuffle reduction. Every one of the 20 random
replays matched all `128*4*64` per-group int32 values exactly for every split.
Final FP32 differences are only reduction-order changes.

| K split | sdot reference | MFMA | reference/MFMA | max abs | max relative L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.555 us | 244.930 us | 0.096x | 1.465e-3 | 2.530e-7 |
| 2 | 23.554 us | 123.864 us | 0.190x | 7.324e-4 | 1.918e-7 |
| 4 | 23.547 us | 81.346 us | 0.289x | 5.188e-4 | 1.590e-7 |
| 8 | 23.538 us | 95.123 us | 0.247x | 4.883e-4 | 1.296e-7 |

These are seven-round ABBA medians with 100 iterations per leg and 20 warmups.
The best split4 candidate is 3.45x slower than sdot, whereas continuation
required MFMA to be at least 2x faster. A single wave has a long dependency
chain of 1024 dynamic MFMA instructions per K4096 tile; splitting shortens that
chain but the extra waves, LDS partials and barrier cannot approach the current
well-distributed dot4 implementation. Split8 is already beyond the optimum.

Therefore this exact no-padding lane mapping is correct but not performant on
CDNA2. Do not build a full routed gate/down kernel from it and do not add a
production selector. The standalone micro is retained as an ISA/layout oracle.

Reproduction:

```bash
amd-smi process --general --sort-by-pid -g 0 1 2 3 4 5 6 7
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/probe_gfx90a_mfma_i8_4x4.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/bench_gfx90a_mfma_i8_gate_tile.py \
  --warmup 20 --iterations 100 --rounds 7 --correctness-replays 20 \
  --output .agents/memory/gfx90a_mfma_i8_gate_tile_oracle.json
```
