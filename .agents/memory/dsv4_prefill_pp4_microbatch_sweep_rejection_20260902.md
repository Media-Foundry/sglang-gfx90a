# DSV4 PP4/TP1 prefill microbatch sweep (2026-09-02)

## Purpose

The first PP4 screen used one 2304-token request per pipeline slot.  This sweep
jointly changed `pp_max_micro_batch_size`, the global chunk token budget, and
`prefill_max_requests`; changing only the first flag cannot change the actual
batch shape.

All runs used four physical gfx90a GCDs (4--7), original weights, PP4/TP1/EP1,
no expert A2A, AIter CKTile A16W4 `KSPLIT=2`, and 32 distinct real code prompts
containing 73,724 server-audited input tokens.  France correctness passed after
the PP4 W2 row-layout fix.

## Pipeline interpretation

`PP_MAX_MICRO_BATCH_SIZE` is the request limit inside each PP ring slot, not the
number of slots. `PP_ASYNC_BATCH_DEPTH=4` makes the ring size eight and hides
last-stage output/CPU work; it does not execute multiple GPU batches on one
stage.  The original 2238 tok/s M2304 result was already a filled four-stage
pipeline.  TP1 makes an individual layer about four times more expensive than
its TP4 shard, so four stages recover approximately one TP4 throughput stream.

For 32 requests the ideal fill fractions are approximately 32/35 for MB1,
16/19 for MB2, and 8/11 for MB4.  Larger microbatches therefore need substantial
single-stage efficiency gains merely to pay for pipeline fill/drain.

## Results

Actual service logs verified the requested batch shapes:

| requests/slot | token budget | observed shape | C32 rounds (input tok/s) | warm result |
|---:|---:|---|---|---:|
| 1 | 2304 | `bs=1,toks=2304` | 2224 / 2238 / 2238 | 2238 |
| 2 | 4608 | `bs=2,toks=4608` | 2369 / 2669 | 2669 |
| 4 | 9216 | `bs=4,toks=9216` | 2078 / 2516 | 2516 |

MB2 raised steady stage markers to roughly 3.15--3.22k tok/s, but after the
pipeline bubble remained slightly below the accepted TP4/EP1 C32 result of
about 2.74k. MB4 stage markers reached roughly 3.9--4.25k but its shorter
eight-microbatch pipeline and ragged tail reduced aggregate throughput.

Cross-round first-token bitwise equality was false in all PP profiles, while
the semantic oracle remained correct.

## Decision

Reject PP4 as the production C32 profile.  Do not scan MB8/MB32: their pipeline
fill fractions are only about 57% and 25%, requiring implausible 60% and 266%
stage-efficiency improvements over MB1.  Return to TP4/EP1 and attack the
large-prefill routed-MoE weight scans directly.
