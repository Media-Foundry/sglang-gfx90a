# TP4 M64 packed-weight-only GLC oracle (rejected)

Date: 2026-08-30

## Question and ISA contract

The strict TP4/EP1 M64 routed path was tested with L1 bypass on packed FP4
weights only.  CDNA2 defines `GLC=1` on a vector read as missing/invalidation
of L1 and fetching from L2; `SLC=1` selects streaming/bypass behavior at L2.
Consequently the candidate sets GLC but never SLC, preserving potential L2
reuse.  Activation, activation scale, E8M0 weight scale, sorter metadata,
router weights and stores retain the default cache policy.

Oracle-only files:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_weight_glc_m64_oracle.cuh`
- `scripts/rocm/bench_dsv4_tp4_m64_weight_glc_oracle.py`

The helper uses explicit `flat_load_dwordx4 ... glc` requests followed by an
`s_waitcnt vmcnt(0)` in the same opaque asm block.  This is deliberate: clang
does not model the VMEM completion dependency of an inline-asm output.  A
source-level cache hint was not accepted as proof.

## Static HSACO proof

The exact cached M64 specializations were extracted and disassembled from
their `.hip_fatbin` sections with ROCm llvm tools.  Artifacts for this boot are
under `/tmp/m64-glc-hsaco`.

| kernel | 128-bit loads | target loads with GLC | SLC | VGPR | SGPR | LDS | scratch | waits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate default | 21 | 0 | 0 | 94 | 51 | 1 KiB | 0 | 122 |
| gate weight GLC | 21 | 4 | 0 | 91 | 52 | 1 KiB | 0 | 105 |
| down default | 19 | 0 | 0 | 50 | 31 | 1 KiB | 0 | 106 |
| down weight GLC | 19 | 2 | 0 | 48 | 37 | 1 KiB | 0 | 100 |

The four gate instructions are exactly the paired gate/up packed rows for the
two R2 output rows.  The two down instructions are exactly the paired logical
W2 packed rows.  No activation, scale, metadata, or output VMEM instruction
has GLC.  The candidate adds neither scratch nor LDS and does not increase the
static wait count.

## Correctness

The real heterogeneous M64 recorder was used:

```text
/tmp/expert_distribution_recorder_1788072257.651073.pt
pass 20, layer 34, TP4
146 active experts, 384 assignments, 174 A4 scans
```

The four arms were:

```text
A default gate + default logical-scale W2 down
B GLC gate + default down
C default gate + GLC down
D GLC gate + GLC down
```

The first independent run passed 100 mutated activation/router-weight cases
bitwise at intermediate BF16, quantized values/scales, FP32 partial and final
BF16, and passed 1000 HIP graph mutation replays bitwise.  The later
shared-output-buffer rerun repeated 100 eager and 1000 graph replays exactly;
graph-private quant allocator tensors were validated through the exact
downstream FP32 partial.

## Timing and decision

Seven rounds used the requested `A/B/C/D/D/C/B/A` order.  The first complete
run gave:

| profile | full routed trimmed mean | delta from A |
|---|---:|---:|
| A | 1013.708 us | -- |
| B | 1111.472 us | +9.64% |
| C | 955.875 us | -5.71% |
| D | 1096.634 us | +8.18% |

The machine was externally noisy, so a second run shared every output buffer
between arms to remove allocation/address-set effects.  Its full-stage raw
CV was 19.6--35.7%; medians were A `986.601`, B `1029.076`, C `1001.759`, and
D `1009.190 us`.  Trimmed means were A `996.265`, B `1121.911`, C `1052.015`,
and D `1091.680 us`.  Thus both-GLC was again slower, by 2.29% at the robust
median and 8.74% at the requested trimmed estimator.  The complete logs are:

- `/tmp/dsv4_tp4_m64_weight_glc_oracle.log`
- `/tmp/dsv4_tp4_m64_weight_glc_oracle_shared2.log`

Although the noisy isolated down-only arm occasionally benefited, it was not
stable across allocation-controlled runs, and the combined policy failed the
required 3% full-routed gain in every run.  Gate-only also regressed rather
than staying within the 2% guard.  This agrees with the earlier M32 result
(`421.546 -> 427.105 us`, 1.30% slower for both GLC).

Reject selective packed-weight GLC at M64.  Do not connect a production
selector, do not add SLC, and do not repeat unless the addressing/load
instruction can remain byte-for-byte identical except for cache bits and a
new controlled platform run eliminates the large external timing variance.

