# DSV4 HIP online-preshuffle oracle and synchronous service rejection (2026-09-02)

## Motivation

The C1/decode production kernels require original checkpoint-order FP4
weights, while AIter CKTile's higher-occupancy prefill kernel requires its
A16W4 preshuffled layout.  Keeping both complete routed-weight copies costs
about 32 GiB/GCD and is incompatible with the target KV capacity.  This oracle
tests a reusable one-layer conversion buffer instead.

## HIP conversion kernel

The new HIP kernel converts the exact TP4 DSV4 tensors:

- W13 packed FP4 and E8M0 scales;
- W2 packed FP4 and E8M0 scales;
- the established gfx90a CKTile W2 inverse 16-row-block permutation is folded
  into the same pass;
- each weight thread moves one aligned 16-byte KPack;
- outputs are written into caller-owned static buffers.

All four outputs are byte-for-byte equal to AIter's
`shuffle_weight_a16w4`/`shuffle_scale_a16w4` plus the W2 row fix.

Physical GCD4, seven-round ABBA:

| implementation | trimmed time/layer |
|---|---:|
| PyTorch/AIter permutation | 5037.99 us |
| HIP, 104 blocks | 3647.57 us |
| HIP, 208 blocks | 2724.63 us |
| HIP, 416 blocks | **2481.21 us** |
| HIP, 832 blocks | 2593.40 us |
| HIP, 1664 blocks | 2551.96 us |

The selected 416-block geometry is 50.7% faster than the existing conversion.

## Layer-ahead overlap oracle

At M4608/E256/Top-6/H4096/I512, original raw weights were converted into a
second static buffer while the current preshuffled buffer ran AIter CKTile:

```text
AIter compute only:          23465.97 us
HIP preshuffle only:          2501.40 us
sequential conversion+MoE:   25867.02 us
two-stream layer-ahead:      22042.95 us
```

The stable two-stream center hides the entire conversion and is about 6.1%
below the compute-only timing.  The first overlap sample was a cold outlier and
was removed by the predeclared seven-round trimmed mean.

## Synchronous production screen

A default-off temporary selector synchronously converted every M4608 layer and
then called AIter.  It preserved the France semantic oracle but regressed the
real 32-request, 73,724-token service workload:

```text
2330.22 / 2292.47 / 2399.02 aggregate input tok/s
median: 2330.22 tok/s
```

The accepted raw MFMA64 queue-aware profile is roughly 2.74k tok/s.  Per-forward
markers similarly fell to about 2.37--2.42k versus the raw profile's roughly
2.9k.  The synchronous production selector and environment variable were
removed; decode and C1 production paths remain untouched.

## Decision

Keep the byte-exact HIP converter and standalone overlap oracle.  Do not wire a
synchronous per-layer crossover.  The only justified continuation is true
layer-ahead ping-pong scheduling: convert layer L+1 on an auxiliary stream while
layer L consumes the other buffer, publish an event before the next routed MoE,
and retain raw weights for all decode/small-M calls.  That integration must
first demonstrate correct buffer ownership across all 43 layers and then pass
France, diverse C1/C32 ABBA, and native-AR decode regression.
