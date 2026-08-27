# gfx90a A4 LDS-sdot packed-weight vector-load HSACO audit

Date: 2026-08-27

## Question

Determine whether the current A4/R2/W8/B832 mode-2 packed-FP4 gate/down
kernels issue eight independent 16-bit weight loads per row/group, and only
build an explicit aligned `uint4`/dwordx4 oracle if that changes VMEM traffic.

## Source geometry and alignment

The source expresses each packed group32 as an unrolled loop of eight
`uint16_t` loads. For mode 2, gate/up uses `gate_base + j*2` and
`up_base + j*2`; down uses `weight_base + j*2`. Every row stride is a multiple
of 16 bytes and every group advances by exactly 16 bytes:

- gate/up K4096 packed row stride: 2048 bytes;
- down TP8 K256 packed row stride: 128 bytes;
- group stride: 16 bytes;
- PyTorch weight storage has stronger-than-16-byte base alignment.

Thus clang is free to combine each unrolled group into one aligned 128-bit
load.

## HSACO result

The current gfx90a JIT objects were extracted from `.hip_fatbin` with
`llvm-objcopy` plus `clang-offload-bundler`, then disassembled with
`llvm-objdump -d`.

Gate/up mode 2 (`E256/M32/T6/I256/K4096/A4/R2/W8/B832`) contains, at the first
weight group:

```text
v_lshl_add_u32 ...                 # gate packed address
v_add_u32 ..., 0x80000, ...        # paired up-row packed address
global_load_dwordx4 v[70:73], ...  # all 16 gate bytes
global_load_dwordx4 v[74:77], ...  # all 16 up bytes
```

The `0x80000` separation is `I256 * packed_K2048`, proving these are the gate
and up weight rows rather than the 32-byte activation read. The following
instructions extract bytes from those four dwords and perform `ds_read_b32`
lookups from the CTA-local pair LUT.

Down mode 2 (`E256/M32/T6/N4096/K256/A4/R2/W8/B832`) similarly contains:

```text
global_load_dwordx4 v[10:13], ...  # complete 16-byte packed weight group
... byte extraction from v10:v13 ...
ds_read_b32 ...                    # pair-LUT decode
```

The separate pair of dwordx4 loads at offsets 0 and 16 in the dot section is
the 32-byte INT8 activation group, not eight scalar weight loads. Scale reads
remain byte/word VMEM operations as expected and are unrelated to packed
weight vectorization.

Whole-kernel instruction inventories include other metadata and activation
accesses, but are consistent with the local proof:

| kernel | `global_load_dwordx4` | `global_load_dword` | narrow global loads |
|---|---:|---:|---:|
| gate/up | 21 | 9 | 2 ushort |
| down | 19 | 19 | 2 ubyte |

## Decision

The requested explicit `uint4` variant would compile to the packed-weight VMEM
instruction already present: one aligned `global_load_dwordx4` per row/group.
It cannot reduce eight weight VMEM instructions to one because clang has
already done so. At best it would reproduce the same load; at worst it would
increase live VGPRs or obstruct the existing scheduling of scale, LUT and
activation reads.

Therefore no vector-load oracle, GPU correctness run or ABBA was created. This
direction is stopped before implementation, as required when HSACO proves the
compiler already vectorizes it. Further work on this kernel must target work
distribution, decode/LUT scheduling or producer-consumer fusion rather than
packed-weight load width.

Reproduction outline:

```bash
llvm-objcopy --dump-section .hip_fatbin=/tmp/fatbin <jit-module.so>
clang-offload-bundler --type=o --unbundle --input=/tmp/fatbin \
  --targets='hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-' \
  --output=/tmp/device.hsaco
llvm-objdump -d /tmp/device.hsaco | \
  rg 'global_load_(dwordx4|dword|ushort|ubyte)|ds_read_b32'
```
