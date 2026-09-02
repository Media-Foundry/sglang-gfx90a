# DSV4 gfx90a CK raw-FP4 logical-B layout audit (2026-09-02)

## Accepted address-level result

The CK logical-B loader must bridge two different physical layouts:

- CK requests BF16 vectors in its `(16,16)` B-preshuffle order.
- Runtime DSV4 FP4 weights and E8M0 scales have already been transformed by
  AIter's `shuffle_weight_a16w4(..., gate_up=true)` and
  `shuffle_scale_a16w4(..., gate_up=true)`.

The runtime oracle now models CK stage 1 as two logical `N=512` buffers that
share one interleaved AIter allocation and select `projection=0/1`.  On physical
GPU 4 it passed 10 random mutations with 4096 independently sampled `bhalf8`
offsets per mutation, bitwise exactly against dequantize-then-preshuffle.

## Full-stage status

The first full M8192 stage attempt with the corrected address mapping still
raised a device illegal-address fault.  This is confined to the default-off
experimental raw-loader instance; the production JIT module was restored after
each attempt.  The standalone oracle proves FP4/E8M0 decoding and gate/up
projection addressing, so the remaining fault is in the CK full-stage buffer
contract (descriptor offsets, expert base, or call ABI), not in the arithmetic
mapping.  Do not enable this path in service until the full routed-stage test is
stable and is compared against the accepted BF16-CK output.

## Current performance baseline (unchanged)

- C1 warm prefill: approximately 2.59k input tok/s.
- C32 real heterogeneous requests: warm median approximately 5.74k input tok/s,
  best observed approximately 6.08k input tok/s.
- Native-AR France correctness baseline: approximately 58 tok/s.

No E2E performance claim is made for the raw-FP4 CK experiment yet.
