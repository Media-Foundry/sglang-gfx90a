# Current-main query16/runtime-M/mix8 C16x8K profile

New diagnostic after accepted mix8, not reuse of the pre-mix8 `capture-v2`.
Original V4, TP8/EP1/no speculation,1M logical KV,32K chunk,16 real code
requests,131069 input tokens, zero prefix hit. Config20 and wide-C4 explicitly
off. Actual mix8/query16/runtime-M hits validated on every rank. Historical
source hashes and all changes to current main are recorded in the plan.

One warmup plus three measured waves,128 rank-forward snapshots (32 warmup).
All64 input echoes match. Three instrumented client wave times:
18.873729,18.867079,18.870353 seconds. This is diagnostic, not a new
uninstrumented throughput checkpoint. Service PID1467721 stopped cleanly;
exec95341 exited0 before the next GPU component test.

Each forward selects the rank with the longest outer envelope and keeps
all of that rank's spans. GPU warm mean wave envelope18.777239s, maximum
rank-envelope spread.68656ms. Event/marker ratios1.000020–1.000042.
This closes the selected envelopes but is not a synchronized multi-rank
critical-path reconstruction. Nested detail spans below must not be added
again to their coarse parents.

| Detail | seconds/wave |
| --- | ---: |
| Routed MoE full stage | 4.389426 |
| Main sparse attention | 2.771669 |
| Both MHC pre-mix | 2.703728 |
| Attention output projection + collective | 1.586810 |
| Indexer logits + adjacent metadata | 1.406534 |
| Both MHC post | 1.371443 |
| MoE output collective (nested) | .754285 |
| Core compressor | .677927 |
| Indexer query | .633195 |
| QKV projection | .594863 |
| Both weighted norm | .570902 |
| Indexer Top-K + metadata | .137140 |
| Indexer compressor | .117646 |
| Both Sinkhorn | .068258 |
| Indexer weights | .028066 |

Do not continue budgeting the old3.06s pre-mix as unoptimized: mix8 now
measures2.70s. MoE and main attention remain substantial. Small Sinkhorn
numbers here belong to the multi-request8K path, not the batch1 FP16 fused
tail in C16x32K; the config20/comb-refinement budget cannot be inferred by
mixing those two profiles.

Evidence directory:
`.agents/experiments/dsv4_c16_final_profile_20260915/capture-mix8-current/`.
Packaged165 files,3,484,153 bytes in `capture-mix8-current-evidence.tar.gz`,
SHA256 `780fa715115cc39a85bd7065c4b1a053b15e0968ae6b433efcba1834836036f9`.
Old accepted and failed captures remain untouched. Analysis enforces128
frames,43 layers, actual per-rank shape agreement, warmup exclusion and
envelope closure before writing the stage results.
