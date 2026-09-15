# Corrected sinks + deterministic CK + H16: 8672.80 input tok/s

Original DeepSeek-V4-Flash, TP8/EP1/no-A2A/native AR, original checkpoint,
1,048,576 logical KV tokens, chunk32768, C16x8K distinct real public-code
requests, zero prefix hits. Sink correction bfce077472 is present throughout.
No DSpark, target pruning, weight-precision changes or KV-capacity reduction.

Previous turn was progress: it repaired the TP sink mapping and reproduced
8376.88 on the corrected H8/atomic baseline. This turn verifies the previously
independent unique-store CK and H16 candidates together on that corrected model.

## Remaining drift before this experiment

The corrected atomic baseline's two C16x128-token waves differed only on request4
(model-loader code), first differing output offset35 after the identical prefix
"...Let me analyze the key issues:\n\n##". One chose "Facts Visible...", the other
"Technical Analysis...". This is an observed divergence location, not a logit-
margin measurement or proof that the first numerical error occurred there.

Unique-store CK retains per-expert FP32 partial arithmetic but performs the
Top6 sum in a fixed slot order, unlike production FP32 atomic accumulation.
It does not promise the same greedy trajectory as atomics. In fact only3/16
whole128-token responses match the previous corrected atomic quality wave.
The separate pre-correction CK investigation already located atomic numerical
variation; this experiment tests the remedy after the sink fix, not its first
component validation. See dsv4_ck_unique_store_20260916.md for the >2GiB invalid
store masking repair, real-data partial oracle and workspace contract.

## Correctness gates

1. New real layer20 fixture comes from the corrected sink service, not the old
   all-ranks-rank0-sink fixture. Every rank has its correct distinct8-head sink.
   Ten direct-HIP peer replays/rank additionally perturb Q and sinks differently
   on each rank; both KV banks remain pair-consistent. All80 results match the
   H8 reference byte-for-byte, initial output matches capture, NaN poison clears,
   and explicit peer close completes.
2. Fresh service with unique CK AND H16: all41 eligible C4/C128 layers,8ranks,
   four large forwards =1312 byte-exact attention comparisons. Per-pair logical
   KV/selection audits agree. Pure-SWA layers0/1 intentionally stay unchanged.
   All8ranks log the verified unique CK module. France passes;1M pool verified.
3. Timing arms disable reference/audit work. Four128-token continuation waves
   per process,16 real requests, all prompt echoes/counts/decoded text verified.

## Formal H8/H16 ABBA, unique CK common to both

Three warm waves per leg;131069 input tokens/wave,16 requests,one output token.
Each arm uses a fresh process except adjacentB1/B2 share their candidate process.
All common runtime source hashes and input manifests agree; no source edits
during the run. Q staging,sink exchange,both pair fences are inside E2E timing.

| Leg | Median input tok/s |
| --- | ---: |
| A1: corrected H8 + unique CK |8363.537057|
| B1: corrected H16 + unique CK |8676.795380|
| B2: corrected H16 + unique CK |8668.806715|
| A2: corrected H8 + unique CK |8373.421849|

Control center8368.479453; candidate8672.801047; **+3.63652%**.
Control A2/A1 drift+0.11819%. This establishes the H16 gain on corrected sinks
and fixed-order CK, without relying on the earlier8664 pre-correction result.
Unique H8 is close to the previous corrected atomic8376.88, but these numbers
alone are not a new unique-vs-atomic ABBA or a tiny speed claim.

## Continuation reproducibility and limits

All three processes each produced four matching waves:16/16 exact per wave.
Across ALL12 waves and both H8/H16 configurations, every request has just one
distinct128-token output.192 completed requests,24576 output tokens; no failed
or empty responses counted as agreement. Request IDs were mapped explicitly,
and prompt token arrays checked before output comparisons.

Reviewed all16 candidate final-wave texts: coherent and source-related, no
obvious garbling/loop collapse.128-token excerpts are not a correctness oracle
for every source-review claim and can end mid-sentence. Reproducibility is not
semantic correctness. No arbitrary-batch or full-logits invariance is claimed.
The previous request4 divergence did not recur in this bounded test.

## Deployment status and cost

Candidate remains an explicit experimental profile. No global launcher default
was changed. validated-launcher.sh contains the testedB startup script plus its
five explicit outer lifecycle environment settings,
not a portable binary distribution. It requires the two local, hash-verified
build manifests for direct-HIP IPC and unique-store CK. It starts the original
model on all8GCDs at localhost:30021 and must not run alongside another workload.

H16 adds512MiB direct-HIP Q/O storage per rank atM32768. Unique CK still needs
six-slot FP32 partial scratch (~3GiB); it is not a capacity reduction. Both fit
the actual1M logical pool in this experiment. Hot runner replacement and broader
length/prefix/admission coverage remain unvalidated; do not silently enable this
for other deployment shapes. Decode remains on the existing native AR path.

Artifacts: .agents/experiments/dsv4_corrected_h16_unique_20260916/.
All measurement services stopped by owned lifecycle; final8GCDs free.

## Updated non-scoring service profile

A fifth fresh process uses the same combined configuration with the existing
asynchronous64-slot GPU markers. Four waves, first excluded;128 rank/forward
snapshots. Each forward selects the longest complete rank envelope and uses
that same rank for every subdivision, never independent per-stage rank maxima.
All8ranks have43 coarse-valid layers and matching input shapes. GPU-event /
realtime envelope ratios1.000337..1.000364; max rank spread4.33ms per forward.

Three HTTP diagnostic waves15.1418/15.1461/15.1442s; mean selected GPU envelopes
15.0462s. These are diagnostic timings, not additional formal speed samples.

| Range | Mean seconds/wave |
| --- | ---: |
| Routed MoE alone |4.3964|
| All MHC/Norm boundaries |3.7656|
| Main sparse attention incl. peer boundaries |1.3829|
| Attention output projection + collective |1.5393|
| MoE output collective |0.7420|
| Indexer query producer |0.6273|
| Indexer owner chain |0.3011|

Nested values are not additive with their coarse parents. MHC breakdown:
post1.3724s,mix1.7287s,Sinkhorn0.0655s,weighted norm0.5717s. First-layer pieces
and boundary gaps account for the difference from the whole MHC envelope.
Complete coarse stages plus interlayer and outer gaps close the15.0462s envelope.

Main attention's old2.7618s budget predates both stages1 AND peer execution;
do not attribute its full reduction to H16 alone. Routed and MHC are now about
29% and25% of the envelope; main attention about9%. The repeated indexer scoring
is no longer a primary budget. Profile service also stopped cleanly.

## Next work

- Keep this corrected, repeatable8672.80 checkpoint distinct from old atomic and
  wrong-sink results. Package the two experiment-only build/ownership contracts
  before considering a production default; preserve explicit fallback.
- Extend fixed-token/long-output checks to different batch placements and prefix
  lengths before claiming whole-model batch invariance.
- The full-stage profile above is now current. Use real-input hardware counters
  to distinguish CK supply/MFMA/writeback costs before a new MoE implementation.
- MHC still merits structural work, but the serial-column post-to-projection
  partial prototype already failed largeM speed AND residual exactness; see
  dsv4_c16_mhc_two_pass_20260915.md. Do not simply rerun that design or blame
  its slowdown on spill (reported spill count was zero).
