# TP8 DSpark C4 Q2×H8 overlap screen (2026-09-10)

## Scope

- TP8 / EP1 / no A2A, strict DSpark gamma=3 target M128.
- 32 heterogeneous open-source code prompts from
  `/tmp/dsv4_open_code_pd_20260908/decode.json` (511/512 input tokens).
- Layer 20, compression ratio 4, eager diagnostic capture only.
- Production weights and production refined C4 attention selection.

## Fixture correctness fix

The eager fixture writer incorrectly rejected negative ragged sentinel slots even
though every production sparse-decode kernel skips `slot < 0`.  The writer now
compacts only non-negative physical slots and preserves negative sentinels in the
remapped index stream.  A CPU fixture with indices `[3, -1, 7, -2]` verified the
remap `[0, -1, 1, -2]` and compact physical slots `[3, 7]`.

## Real C4 evidence

Fixture:

`/tmp/dsv4_tp8_q2h8_c4_fixture_l20_long3/layer_20_rank_0_c4.pt`

Across the 64 adjacent target-query pairs:

- Row lengths: 256/256 or 256/257 keys.
- Physical-slot set Jaccard: 0.9884--0.9922 (median 0.9922).
- Exact aligned 16-key tiles: median 8 of 16, but 30/64 pairs had none.
- 34/64 pairs had at least one exact aligned tile.
- Ordered common prefix: zero keys for every pair.
- All 64 pairs had physical-slot Jaccard above 0.5.

The high set overlap is real, but C4/SWA ordering rotates between adjacent
positions.  Therefore the safe-prefix Q2 contract cannot cover this workload:
whole-pair production-order compatibility is 0/64.

## Decision

Do not reconnect the existing Q2 service candidate.  A correct next candidate
must build a union schedule plus per-row membership, or canonicalize both rows;
either changes the online-softmax accumulation order and requires target-logit,
accept/reject, commit-state, France, repetition, and multi-round E2E validation.
The simpler device `pair_safe` + production fallback design would fall back for
all observed C4 pairs and cannot provide a speedup.

The eager run was diagnostic only (graphs disabled): resident throughput was
329.5 tok/s and is not a production performance result.  It completed all 32
requests with 16 tokens each and no service failure after the fixture fix.

