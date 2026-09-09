# TP8 strict DSpark draft-token synchronization guard

Date: 2026-09-10

## Problem and scope

The existing strict TP8 profile synchronized the target accept decision only
after target verification.  A rank-local last-bit difference in the draft
head could therefore produce different draft token IDs on different TP ranks,
which would then enter the same collective target forward.  Synchronizing the
accept result after that forward is too late to make the target input valid.

`SGLANG_DSPARK_SYNC_DRAFT_ACROSS_TP` now broadcasts rank 0's contiguous
`[bs, gamma]` draft-token tensor once, in place, immediately after proposal and
before confidence, layout planning, and target verification.  It is default
off globally and enabled only by the explicit strict TP8 full-target profile.
Native AR is unchanged.

## Validation

- Unit tests cover in-place identity, contiguous payload, authoritative rank-0
  replacement, and launch-profile wiring: 5 passed.
- Fresh TP8/EP1/no-A2A service, original checkpoint, gamma 3, one-million-token
  pool, and 32 heterogeneous real coding requests.
- All eight ranks hit C128 CK attention, AR12, G832/D832 and M128 gate/down
  row-prefetch.
- Resident decode: warm 1107.32 tok/s; measured 1090.63 and 1080.20 tok/s
  (median 1085.42 tok/s).
- 96/96 requests passed the severe repetition gate; worst last-512 duplicate
  8-gram fraction was 0.168.
- GPU processes were absent after the run.

Artifacts: `/tmp/dsv4_tp8_dspark_sync_draft_e2e_20260910/`.

## Remaining limitation

This closes a TP protocol hole but does not make rank-0 floating-point
execution bitwise deterministic across independent waves.  Exact full-output
hashes were 0/32 between each pair of waves; common-prefix medians were 22 and
9 tokens.  The remaining first-divergence investigation must compare fixed
teacher-forced rank-0 boundaries rather than autoregressive final hashes.
