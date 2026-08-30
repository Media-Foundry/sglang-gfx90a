# DSV4 TP4 M64 exact signed-int5 repack oracle (rejected)

Date: 2026-08-30

## Question

Test the remaining point between packed FP4 plus an LDS pair LUT and the
already-rejected fully expanded INT8 weights: repack the exact E2M1 signed
codebook values at load time into five bits per value, then expand them with
SWAR in the hot grouped-down kernel and retain the existing signed INT8
`sdot4` arithmetic and fixed reduction order.

This was an oracle only.  No production selector or model-loader path was
changed.

## Layouts

Both layouts store exactly 20 bytes per group of 32 weights, or 1.25x the
original packed FP4 bytes and 0.625x a fully expanded INT8 cache.

- dense5: 32 consecutive two's-complement int5 fields in five uint32 words;
- bitplane5: four uint32 words containing the low nibbles plus one uint32
  high/sign plane.  The latter avoids fields crossing dword boundaries.

The oracle used the real TP4 M64 route from recorder pass 20/layer 34,
A4/R2/W8/D832, W2 `[256,4096,512]`, group-32 INT8 activations, E8M0 scales,
and the existing routed-weight/fixed-slot partial layout.

## Correctness

- baseline packed FP4/LDS, dense5 and bitplane5 partial FP32 tensors were
  bitwise equal;
- 100 eager mutations of INT8 activations, activation scales and router
  weights remained bitwise equal;
- both int5 layouts passed 1000 HIP graph replays and remained bitwise equal;
- repacked W2 occupied 335,544,320 bytes, exactly 1.25x packed FP4.

An initial dense5 implementation exposed a helper bug: its per-byte sign
extension mapped int5 `31` to `0x8f`, not `0xff`.  The exact form is
`bytes | ((bytes & 0x10101010) * 15)`.  All reported measurements were taken
only after fixing this and passing the mutation/graph suite.

## HSACO resource gate

| layout | VGPR | scratch | weight loads |
|---|---:|---:|---|
| dense5 | 50 | 0 | aligned `global_load_dwordx4` + `global_load_dword` |
| bitplane5 | 60 | 0 | aligned `global_load_dwordx4` + `global_load_dword` |

The remaining `global_load_ubyte` instructions are E8M0 scale loads, not
int5 weight loads.  Both candidates passed the <=104 VGPR/no-spill/no-narrow-
weight-load compile gate.

## Down-only ABBA

Seven A/D/P/P/D/A rounds, 10 warmups and 100 iterations per sample:

| path | median (us) | trimmed mean (us) | versus packed |
|---|---:|---:|---:|
| packed FP4 + LDS LUT | 303.032 | 303.005 | baseline |
| dense5 SWAR | 367.430 | 367.496 | +21.3% |
| bitplane5 SWAR | 407.197 | 407.191 | +34.4% |

The bitplane layout produces cleaner field boundaries but needs 10 additional
VGPR and substantially more bit-spreading instructions.  Dense5 is better but
still loses badly because the 25% extra weight traffic plus SWAR exceeds the
cost of the accepted compact FP4/LDS lookup.

## Decision

Reject and stop.  Do not implement the gate/up half, loader integration or a
production selector.  Together with the prior +43.8% full-INT8 rejection,
this closes exact load-time E2M1 code expansion at both 5-bit and 8-bit storage
points for the current grouped `sdot4` work decomposition.

