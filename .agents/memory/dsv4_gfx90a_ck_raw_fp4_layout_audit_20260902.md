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

The illegal-address fault was a test-composition error rather than a remaining
buffer-layout bug.  The explicit block-M64 stage-1 instance had initially been
combined with the heuristic block-M128 sorter.  Enabling the accepted FP32
stage-2 profile also propagates block-M64 to sorting; with that matched pair the
M8192 full stage completes.

The scalar logical loader is nevertheless decisively too slow:

- balanced routing: approximately 88.97 ms;
- skewed routing: approximately 101.95 ms;
- accepted materialized-BF16 CK path: approximately 12 ms at M8192.

There was also small replay drift (balanced max-abs 128; skewed max-abs 256),
so this implementation is rejected for service.  A future attempt must replace
eight scalar address/decode operations with aligned packed loads, `v_perm`
E2M1 expansion, and one reused E8M0 scale per K32 group.  It should not proceed
to E2E unless the complete routed stage is both stable and at least 5% faster
than materialized BF16 CK.

## Current performance baseline (unchanged)

- C1 warm prefill: approximately 2.59k input tok/s.
- C32 real heterogeneous requests: warm median approximately 5.74k input tok/s,
  best observed approximately 6.08k input tok/s.
- Native-AR France correctness baseline: approximately 58 tok/s.

No E2E performance claim is made for the raw-FP4 CK experiment yet.
