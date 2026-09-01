# DSpark M128 progressive boundary checkpoint (2026-09-01)

## Scope

This is a standalone TP4/gfx90a gamma-three target-verify boundary oracle. It
does not alter native AR, model weights, or the production DSpark service.
Physical GCDs 4--7 were used after `amd-smi` reported no processes.

The baseline overlaps the current TP4 shared M128 expert with the real routed
M32 FP4 path, joins anchor rows, performs the production AIter M128 all-reduce,
and then consumes the draft M96 rows with four BF16 projections plus the CK
sparse-attention kernel. The candidate exposes the three draft rows per request
from one logical M128 collective epoch while routed M32 is still running, then
publishes and reduces the 32 late anchor rows in the same epoch.

## Correctness bug and fix

An initial split implementation issued the anchor-ready system release from an
empty synchronization kernel after a different producer kernel updated the
anchor rows. On CDNA2 that release did not reliably publish writes made by
unrelated workgroups to peer GCD readers. After isolating the baseline and
candidate registered buffers, their pre-AR inputs were bitwise identical while
only the late anchor output differed, proving the defect was the publication
protocol rather than graph aliasing.

The fixed candidate uses one CTA to fuse:

```text
shared anchor partial + routed M32 partial
    -> BF16 joined anchor
    -> GLC buffer writeback
    -> system-scope release to every rank
    -> peer-read anchor reduction
```

The workgroup that writes the payload now also publishes the release. GLC peer
loads and the production owner-rotated FP32 accumulation order are retained.

The composed graph is bitwise exact on every rank for:

- the complete M128 output;
- the compact M96 draft snapshot;
- all four BF16 projection outputs;
- the CK sparse-attention output.

## Seven-round ABBA

Context 256, warmup 10, 30 graph replays per leg, rank-max timing:

```text
baseline median:     779.641 us
progressive median:  436.782 us
net saving:          342.858 us/layer
gain:                 43.976%
```

Both A sides and both B sides were stable, and the result clears the pre-set
100-us/layer continuation gate by a wide margin.

## Production constraint

This is not an E2E result. The next layer's full sparse attention cannot be
blindly launched before its anchor KV is available. Production integration
must remain strictly TARGET_VERIFY, TP4, BS32, M128, gamma-three and may advance
only row-local draft work (MHC/norm and projection producers) before the anchor
join. The attention consumer must retain its causal dependency on anchor KV.
Native AR must remain unreachable from the new path.

Files:

- `gfx90a_tp4_m128_progressive_ar_oracle.cuh`
- `gfx90a_tp4_m128_progressive_ar_oracle.py`
- `bench_dsv4_tp4_m128_progressive_ar_oracle.py`
- `bench_dsv4_tp4_m128_progressive_boundary_oracle.py`
