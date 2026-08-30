# DSV4 TP4 M32 accepted A4 HSACO/cache audit

Date: 2026-08-30

## Exact production specializations

The TP4 throughput profile selects the following strict M32 path:

- gate/up: A4/R2/W8/G2080, LDS pair LUT, DPP reduction;
- down: A4/R2/W8/D832, row prefetch, logical W2 scales;
- fixed FP32 partial and deterministic reduction are unchanged.

The audited code objects were the corresponding cached JIT specializations,
not the generic grouped fallback. The `.hip_fatbin` section was extracted with
`llvm-objcopy`, unbundled for
`hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`, and disassembled with ROCm
`llvm-objdump --mcpu=gfx90a`.

## Resources and instruction inventory

| item | DPP gate/up | row-prefetch logical-scale down |
|---|---:|---:|
| VGPR | 95 | 50 |
| SGPR | 51 | 31 |
| LDS | 1024 B | 1024 B |
| private/scratch | 0 | 0 |
| wavefront | 64 | 64 |
| max workgroup | 512 | 512 |
| `global_load_dwordx4` | 21 | 19 |
| `global_load_dword` | 9 | 13 |
| narrow global loads | 2 `ushort` | 2 `ubyte` |
| global stores | 8 `short_d16_hi` | 8 `dword` |
| `ds_read_b32` | 64 | 32 |
| `ds_bpermute_b32` | 32 | 32 |
| `v_dot4c_i32_i8` | 128 | 64 |
| `s_waitcnt` | 77 | 100 |

All 40 gate and 42 down VMEM instruction lines use the default cache policy:
there is no `glc`, `slc`, or other non-temporal modifier. Packed weights are
already emitted as aligned `global_load_dwordx4`; the earlier vector-load audit
was correct and explicit wider C++ loads cannot reduce their instruction count.

Gate waits contain 12 `vmcnt(0)`, 10 `vmcnt(1)`, and 10 `vmcnt(2)`. Down has
18/11/9 respectively, plus one each of `vmcnt(3)` and `vmcnt(4)`. Thus the
accepted row-prefetch schedule already carries several VMEM operations across
dot/decode work; it is not the rejected immediate-wait scalar pattern.

## Existing prefetch/cache results that must not be repeated

- Gate K-group distance-one prefetch survived in HSACO and raised VGPR
  94 to 96, but slowed full routed by 1.36%.
- Gate R2 same-group row prefetch improved the isolated gate, but the all-layer
  saving implied only about 0.33% of the model step and service was flat.
- Down R2 row prefetch is already in the accepted path.
- CTA-wide activation/scale LDS staging slowed down by 15.44%.
- Full INT8 pre-expansion removes FP4 decode but doubles weight bytes and was
  43.8% slower in the latest exact TP4 oracle.
- N64 hot W2 cache and gate/up interleaving/repacking were rejected; neither
  should be repeated as a cache-policy experiment.

## New cache/load oracle: L1-bypass packed weights only

The ISA states that `GLC=1` on a vector load intentionally misses/invalidate
L1 and fetches from L2, while `SLC=1` makes L2 streaming/non-temporal. The
accepted kernel currently leaves both unset for every VMEM load.

The untried oracle is therefore **selective weight-only `glc`**:

- issue only packed FP4 weight `global_load_dwordx4` with `glc`;
- keep activation, activation-scale, weight-scale, sorter metadata, and output
  accesses at the default cache policy;
- never set `slc`, so L2 weight reuse remains available;
- preserve the accepted issue order, A4/R2 mapping, LUT decode, SDOT order,
  partial layout, and reduction exactly.

This is different from software prefetch. It does not add VMEM operations or
live prefetch registers. It tests whether one-use packed rows are evicting the
small, heavily reused A4 activation/metadata working set from each CU's L1.
With diverse M32 routing, most expert runs are short and A4 already captures
the useful within-block weight reuse, making weight L1 persistence a plausible
net loss.

Use four standalone profiles in a seven-round `A/B/C/D/D/C/B/A` comparison:

```text
A default
B gate weight-glc only
C down weight-glc only
D both
```

First prove in HSACO that only the targeted packed-weight dwordx4 instructions
gain `glc`, with unchanged VGPR/LDS and no extra waits. Then run 100 input and
router-weight mutations plus graph replay, followed by real diverse and
correlated M32 routes. Continue to service only if D improves full routed by at
least 3%, neither workload regresses by more than 2%, and all intermediate and
final tensors are bitwise exact. A plausible 3--5% saving in each weight-heavy
producer would save roughly 12--20 us from a 420--440 us full routed stage,
crossing the requested 3% gate without consuming VRAM.

## Packed-nibble native-instruction audit

The accepted LDS decoder's static body contains only three `v_perm_b32`
instructions because `v_perm` initializes the 1-KiB pair LUT; hot packed E2M1
decode uses 64/32 `ds_read_b32` instructions in gate/down. Dot execution is
128/64 `v_dot4c_i32_i8`.

CDNA2 does have native `V_DOT8C_I32_I4`, `V_DOT8_I32_I4`, and unsigned U4
variants. It does **not**, however, directly consume E2M1 FP4 times the current
signed INT8 activation:

- E2M1 nibble encoding is not signed two's-complement I4 and still requires a
  LUT/SWAR transform;
- E2M1 values can be represented as signed I4 after moving the existing 0.5
  factor into the scale, but the activation spans signed INT8 and cannot be an
  I4 operand without requantization or multi-digit decomposition;
- exact INT8 decomposition needs at least two I4 dots plus carry/sign fixup for
  each eight elements, removing the apparent 2x dot-density advantage over two
  current I8 `dot4` operations;
- the already tested exact magnitude/sign UDOT8 design needed multiple unsigned
  dots plus SWAR and MXFP4 activation quantization and slowed the routed stage
  by 67.7%.

Therefore a native I4 instruction exists, but there is no direct packed-E2M1
W4A8 instruction path. The only untried signed-I4 decomposition has unfavorable
static instruction economics and duplicates the rejected UDOT8 principle. No
kernel should be written unless the activation representation itself changes;
that would no longer be an exact load/decode-only oracle.

## Weight-only flat GLC oracle result

The proposed oracle was implemented as a standalone JIT specialization and
tested before any production connection. Opaque inline flat VMEM requires an
explicit wait: the first attempt correctly formed 64-bit flat addresses but
clang did not insert `s_waitcnt` before consuming inline-asm outputs, producing
incorrect tensors. The corrected helper issues both R2 packed-row requests and
`s_waitcnt vmcnt(0)` in one asm block. This prevents compiler motion across the
wait and keeps the two row requests paired.

Final HSACO checks:

- gate: exactly four packed-weight loads are `flat_load_dwordx4 ... glc`, no
  `slc`, 91 VGPR, 52 SGPR, 1 KiB LDS, zero scratch;
- down: exactly two packed-weight loads are `flat_load_dwordx4 ... glc`, no
  `slc`, 48 VGPR, 37 SGPR, 1 KiB LDS, zero scratch;
- all activation, scale, metadata and output VMEM instructions retain the
  default cache policy.

On real pass37/layer34 routing (106 active experts, 192 assignments, 113 A4
scans), A/G/D/B passed 100 eager input/router-weight mutations and 1000 graph
mutation replays bitwise exactly at intermediate BF16, quantized values/scales,
FP32 partial and final BF16 output.

Seven-round `A/G/D/B/B/D/G/A`, trimmed means:

| stage | A default | G gate GLC | D down GLC | B both GLC |
|---|---:|---:|---:|---:|
| gate | 246.625 us | 251.977 us | 246.446 us | 252.139 us |
| quant | 40.795 us | 40.912 us | 40.868 us | 40.875 us |
| down | 160.877 us | 160.810 us | 162.143 us | 162.211 us |
| reduce | 3.596 us | 3.628 us | 3.610 us | 3.619 us |
| full routed | 421.546 us | 426.159 us | 422.587 us | 427.105 us |

Both-GLC is **1.30% slower**, rather than at least 3% faster. Gate-only is the
main regression; down-only is also slightly slower in the full chain. The
default L1 policy is therefore beneficial or, at minimum, cheaper than the
explicit flat-load/wait sequence. Reject this direction and do not connect a
selector or repeat it with `slc`.

Artifacts retained outside production were the benchmark log
`/tmp/dsv4_tp4_m32_weight_glc_oracle.log` and extracted HSACO directory
`/tmp/glc-final.IQY6` for the current boot. The experimental source plumbing
was removed after the negative decision.
