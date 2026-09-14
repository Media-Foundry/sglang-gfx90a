# C16 prefill follow-up priorities (2026-09-15)

User review accepted while runtime-M service ABBA is running. This is a work
queue, not a record of completed experiments or new measured gains.

1. Finish the fresh A1/B1/B2/A2 group16 constant-M versus runtime-M run in
   `dsv4_c16_indexer_runtime_service_20260915`. Require identical input IDs,
   zero prefix hits, 1M logical KV, matching timed forward counts, all-rank
   selector evidence and multi-M compilation reuse. Check long outputs
   separately from component bitwise equality. Do not promote group16 or
   runtime-M until serving validation completes.
2. Reuse the asynchronous stage markers on the resulting final configuration.
   Select one longest-envelope rank per forward and use that same rank's
   component intervals. The prior 23.49-second diagnostic predates both MHC
   optimizations and query reuse; its pre-mix/logits costs are not remaining
   optimization budgets. Keep diagnostic timings separate from normal E2E.
3. Screen MHC pre-mix four-row versus eight-row reuse, preserving FP32 Fn,
   K1024 chunks, per-row reduction order and RMS arithmetic. Implement every
   accumulator, ragged mask and store; setting BM=8 alone is not an eight-row
   implementation. Include row permutations and irregular M. Only pursue
   service ABBA after stable roughly 10% full-component benefit on relevant
   shapes; examine registers and occupancy. No large FP32 activation workspace.
4. Independently validate query reuse at 16K/32K and mixed prefixes. Do not
   truncate KV or merely remove the width guard. Record scratch capacity,
   page-sharing rate, valid tile fraction and unchanged logical/physical
   selection. This is coverage work, not a predicted gain on C16 x 8K.

Top-K and Sinkhorn are not priorities absent a new profile showing otherwise.
Native decode, DSpark, V4.1 and TP4 remain outside these selectors.

At this note's creation, production remains query group4. The completed warm
group16 trial was 6877.48 input tok/s versus 6778.97 (+1.45%); it did not
resolve exact-M cold compilation. The runtime-M integrated component test
passed 700 mutations with exact scores and logical/physical Top-K, but the
service trial is still in progress. Whole-model determinism remains unproven.

## In-progress P0 evidence (not an accepted ABBA result)

Fresh A1 service PID1308230, birth identity checked by the owned lifecycle
helper. The control trace now records M32765/32766/32767/32768 on all eight
ranks, with identical width2048/page-stride32/group16/pointer-alignment
metadata and four distinct compiled hashes per rank. `check_shapes(..., 0)`
passes. Its excluded cold warmup measured 2676.06 input tok/s; do not compare
that directly with the previous warm 6877 figure. B and A2 are not yet done.
CPU regression: 25 tests plus 31 subtests passed; four compile-trace checker
tests passed. One unrelated pytest `asyncio_mode` configuration warning.

A1 subsequently completed and stopped cleanly; B launched as PID1317173
(owned birth identity verified). A1 timed rates were 4593.35/6878.88/6873.63;
the first timed wave encountered another M32765 compilation. Its leg median
is 6873.63, but this is NOT three compile-free warm waves. The analyzer now
reports per-leg serving compilation events rather than hiding this condition.

A1's two quality waves matched 15/16. Case4 diverged after33 completion tokens
with identical echoed input IDs. Both visible 128-token excerpts discuss
weight-loader concurrency coherently; one organizes facts, the other starts a
check-then-act analysis. `counter_before_loading_weights` and the named
multi-thread iterator really occur in the supplied prompt. This is a bounded
semantic spot-check, not proof of whole-model determinism or full correctness.

Prepared `dsv4_c16_final_profile_20260915/capture.py` for P1. It has NOT run:
it requires completed ABBA summary, stopped source service, matching source
hashes, all four optimization flags, group16/runtime mode matching the chosen
source arm, TP8 native AR and 1M KV. It reuses the four-forward/one-warm-wave
marker protocol and explicitly disables compile-shape debug printing. AST
validation passed. Final profiling remains gated on P0 completion and review.

B's in-progress trace has now passed `check_shapes(..., 1)` for every rank:
M32766/32767/32768 reuse one compiled hash and one cached Python object per
rank with matching non-M metadata. No serving compilation warning appears
at this observation point. Excluded warmup6115.97 input tok/s is not a steady
result and not a fresh-empty-disk-cache benchmark; the runtime binary had
already been built by the component test. The key claim is no extra variant
for a changed M, not elimination of the first compilation for every layout.
