# First V4 post-to-projection-partial prototype rejected

2026-09-15. Independent component experiment, never dispatched by production.
Current accepted default8K C16 prefill remains8384.698465 input tok/s.

Candidate computes H1024 residual tiles, writes BF16 residual and H256 RMS
partials, and uses the newly rounded residual to produce24 FP32 projection
partials for each of16 K1024 chunks. A second kernel accumulates the16 chunks
in explicit order and applies RMS scaling. No previous-layer coefficient
substitution, no FP16 Fn, no changed checkpoint, no grid spin/barrier protocol.
Scratch is48 MiB atM32768. The implementation retains four old residual vectors
per row/tile and processes projection columns serially.

Baseline is the accepted fused4 post plus reuse8 paired-column pre-mix, NOT the
older unoptimized post/pre. Both measured paths include both relevant kernels.
Inputs repeat real layer0 sampled x/residual/post/comb; Fn is captured FP32.
This is synthetic occupancy expansion of real samples, not a live full-M tensor
or a model-service experiment. Three input mutations per M, three ABBA cycles,
three calls per event sample, physicalGCD4 only with service stopped first.

| M | Baseline ms | Candidate ms | A/B |
|---:|---:|---:|---:|
|17|0.184729|0.075635|2.4424|
|128|0.272388|0.114399|2.3810|
|8192|2.360529|4.323953|0.5459|
|32768|8.886784|17.075466|0.5204|

All cases fail byte equality already at residual, as well as RMS and mixes.
Largest observed M32768 mix absolute difference across three inputs0.01205444.
Therefore even the small-M component advantage is NOT an accepted exact path.
The data do not establish the exact compiler/arithmetic cause of the residual
difference. No claim that it is solely reduction ordering or FMA is made.

Resource metadata for the fused producer:42 registers,0 spills,16 bytes shared.
The large-M slowdown cannot be called a spill regression. Logical Fn loading
loses the baseline's eight-row sharing, a plausible cost, but no hardware
counter diagnosis was collected. This rejects this one serial-column producer,
not every possible V4 two-pass organization. Do not promote, or broaden selectors.

Artifacts: .agents/experiments/dsv4_c16_mhc_two_pass_20260915/
candidate.py, screen.py, result.json. Result includes source/fixture hashes,
all mutations and raw ABBA timings. First whole-model frontier service had
already exited before this GPU test; the second diagnostic service started
only after this test finished. No overlapping GPU experiments.

## One-wave follow-up

After both completed all-layer diagnostic services stopped, screen_w1.py changed
ONLY the candidate producer to num_warps=1. Baseline remains four-wave fused4
post plus one-wave paired pre-mix. Three ABBA cycles again:

- M8192:2.372124 ->3.875498 ms.
- M32768:8.976324 ->15.796143 ms (candidate about76% slower).
- Registers123, spills0, shared0. All residual/RMS/mix byte checks still fail.

Thus removing cross-wave reductions helps part of the first candidate cost but
does not make this design competitive. CPU analysis of the service capture ran
concurrently with this isolated GPU screen; neither run had another GPU workload.
Do not use the small-M launch-sensitive numbers as an accepted speed claim.

Read-only compiler inspection found different scalar/packed FMA generation
between cached post-reference and candidate artifacts. This supports checking
contraction semantics next, but no controlled contraction experiment proved the
precise source of the output difference. The reference cache object was not
independently matched to the current service object in this inspection.
Both prototypes remain experiment-only. No improvement to production claimed.
