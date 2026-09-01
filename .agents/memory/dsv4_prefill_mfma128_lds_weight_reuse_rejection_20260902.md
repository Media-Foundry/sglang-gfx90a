# DSV4 prefill MFMA128 LDS weight-reuse rejection (2026-09-02)

## Hypothesis

At M13824--M16128, each expert receives roughly 324--378 routed assignments.
The production A64 MFMA kernel therefore scans a given expert weight five or
six times. An oracle assigned two independent 64-row wave groups to one CTA and
staged the raw FP4 weight tile once in LDS. Each assignment group used K-split
2, so the CTA retained four waves and the same 64-row accumulator footprint per
wave as production.

## Implementation properties

- Two A64 groups, four wave64 waves total.
- Raw FP4 weights and E8M0 scales staged in LDS for 2/4/8/16 K phases.
- Fixed two-way FP32 partial reduction per assignment group.
- No production selector or model path was modified.

The first compiler result unrolled eight phases and produced VGPR512 plus 136
bytes of scratch per thread. Explicitly preventing phase-loop unrolling removed
all scratch and reduced the resource footprint to:

```text
VGPR 180
SGPR 58
LDS 41472 bytes (8-phase build)
scratch 0
wave64
```

## Correctness

Against the production A64/K-split4 gate/up kernel:

- M2304 max absolute BF16 difference: 6.1035e-5.
- M13824 max absolute BF16 difference: 1.2207e-4.
- All tested mutations remained finite and within BF16 tolerance.

The difference is expected from the K-split2 versus K-split4 FP32 association.

## ABBA performance

At M2304, occupancy is too low for A128: the A64 metadata had 275 blocks while
A128 still had 256. The spill-free 16-phase candidate was therefore much worse:

```text
A64 reference:  6466.0 us
A128 LDS:      15709.6 us
regression:      143.0%
```

The decisive high-occupancy M13824 run had 1423 A64 blocks and 768 A128 blocks:

```text
A64 reference trimmed: 44165.3 us
A128 LDS trimmed:      57236.7 us
regression:               29.6%
```

## Decision

Reject and remove the oracle. Even after block count falls by 46%, LDS staging,
barriers, and VGPR180 cost more than the avoided raw-weight loads. Do not retry
larger assignment tiles, more LDS phases, or a wider accumulator in this kernel
family.

The next high-M route must reuse CK/AIter's spatially local grouped scheduling
while consuming the raw checkpoint layout, or change the loader/layout strategy
without duplicating the full expert weights. It should not add a second explicit
LDS publication protocol around the existing MFMA64 kernel.

