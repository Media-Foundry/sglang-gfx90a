# DSpark causal two-lane rejection (2026-09-01)

## Scope

This experiment corrected an important dependency flaw in the earlier
semantic-lane proposal. Draft target-attention rows cannot run before the
current candidate block's anchor KV exists. Once the complete KV block has
been projected and stored, however, attention query rows are independent. The
tested dependency graph was therefore:

```text
anchor lane: CK attention M32 -> routed expert M32
draft lane:  CK attention M96 -> shared expert M96
```

The production control was:

```text
CK attention M128 -> (routed expert M32 || shared expert M128)
```

All tests used original production shapes on physical GCD 4 after `amd-smi`
reported no processes. The oracle uses synthetic activations/weights only to
measure gfx90a resource contention; it does not alter production model code.

## Preliminary producer split

Adding the four attention producer shapes N=1536/2048/512/64 after the exact
entry-MHC row split gave:

```text
full M128 chain:   266.457 us
draft M96 chain:  254.474 us
anchor M32 chain: 190.596 us
hideable ceiling:  75.861 us/layer
```

MHC state remained row-separable, but M-specific BF16 GEMM solutions made the
projection outputs non-bitwise on 100/100 mutations. After progressive-AR
overhead, this producer-only plan still offered only about 47 us/layer.

## CK M32 screen

The old oracle deliberately routed M32 attention through Triton. Enabling the
same CK/MFMA sparse kernel used by M96/M128 reduced the complete M32 attention
arm (four projections, sparse core and output tail) from:

```text
Triton M32: 440.324 us
CK M32:     292.401 us
```

This justified measuring the complete two-lane schedule, but was not enough to
make it win.

## Seven-round result

```text
causal two-lane candidate:      1011.997 us/layer
matched production baseline:    964.904 us/layer
candidate delta:                 +47.093 us (+4.88%, slower)
```

The production baseline explicitly runs full M128 attention first, then
overlaps routed M32 and shared M128 on separate streams. The candidate's M32
kernel launch/projection fixed costs and concurrent HBM/CU contention exceed
the work hidden by the two row lanes.

## Decision

- Reject the causal row-lane production integration.
- Keep `--ck-m32`, `--anchor-attention-before-moe`, and
  `--production-baseline` as oracle-only diagnostics.
- Do not split target attention solely to expose anchor rows; M128 CK remains
  the better full-block execution shape.
- Native AR and the accepted DSpark service remain unchanged.
