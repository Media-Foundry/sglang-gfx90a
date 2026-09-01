# DSpark M128 split-shared all-reduce rejection (2026-09-01)

## Hypothesis

For gamma-3 target verification at BS32, split the M128 MoE reduction into a
draft-only M96 shared-expert reduction and an anchor M32 routed+shared
reduction. Issue the two collectives in identical rank-global order while
overlapping the first reduction with anchor routed-expert compute.

## Correctness and stability

- Physical GCDs: 4,5,6,7.
- HIP graph capture completed; unlike concurrent full-graph replay, this did
  not deadlock.
- Three independent BS1 France rounds reproduced the historical exact answer.
- Both heterogeneous BS32 rounds completed all 32 requests at 256 tokens.
- Concurrent France was semantic in one of two rounds, so the known BS32
  greedy-trajectory variability remained.

## Performance

The fixed randomized heterogeneous manifest was
`/tmp/dsv4_2k_seed20260901.json` (selected workload SHA-256
`6699bf7e5153eaf6625f72954d8cad92064c944067874a6297bb780cf2958f5c`).

Resident BS32 throughput was 1109.74 and 1100.12 tok/s, versus the accepted
M128 CK checkpoint center of about 1559 tok/s. This is approximately a 29%
regression.

## Conclusion

Reject and remove the production experiment. Two shape-split TP collectives
add enough rendezvous, launch, and synchronization cost to dominate the small
amount of routed M32 compute hidden underneath the M96 reduction. Future
overlap work must preserve a single M128 collective and overlap compute-only
segments before that boundary.
