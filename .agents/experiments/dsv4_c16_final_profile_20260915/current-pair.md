# Accepted paired-column C16x8K: current service profile

Original V4, TP8/EP1/no A2A/native AR, original weights, 1M logical KV,
chunk/max-prefill32768. Query16/runtime-M/mix8/paired columns on; wide-C4,
config20 and comb-refinement off. This is a diagnostic, not a new performance
checkpoint replacing the uninstrumented7367.6734 input tok/s ABBA.

Fresh process1567196; one warmup and three measured waves,16 real8K code
requests per wave,131069 input tokens. All64 exact input echoes and zero cache
hits verified. All eight ranks log actual pair8x2 and query16/runtime-M paths.
Capture exec3863 exited0 and the owned process tree stopped with remaining=[];
all GPUs were confirmed free before any subsequent experiment.

Instrumented measured client wall times17.82629945 /17.81990152 /17.81415382s.
128 rank-forward frames,43 layers each. Exclude all32 warmup frames. For each
forward choose the rank with the longest outer envelope and retain its stages.
Mean warm GPU envelope17.726924s; maximum rank-envelope spread4.2808ms;
event/marker envelope ratios0.99998575..1.00001277. This is not a fully
synchronized multi-rank critical-path reconstruction; nested detail spans must
not be added to their coarse parents.

| Detail | seconds/wave |
| --- | ---: |
| Routed MoE full stage |4.379018|
| Main sparse attention |2.768829|
| Both MHC pre-mix |1.730476|
| Attention output projection + collective |1.577763|
| Indexer logits + adjacent metadata |1.404546|
| Both MHC post |1.361075|
| MoE output collective, nested |.751398|
| Core compressor |.674849|
| Indexer query |.628421|
| QKV projection |.577319|
| Both weighted norm |.571059|
| Indexer Top-K + metadata |.136824|
| Indexer compressor |.117257|
| Both Sinkhorn |.065525|
| Indexer weights |.027978|

The preceding mix8 profile measured2.703728s pre-mix and18.777239s GPU envelope.
The new pre-mix is approximately.973s shorter; envelope approximately1.050s
shorter. These are separate diagnostic captures, not component ABBA attribution,
but their scale agrees with the independent5.83% serving gain. Do not count the
old2.70s pre-mix as remaining budget.

Capture plan records all source hashes and the only changed historical source:
launcher default promotion, with identical explicitly selected kernel flags.
The old `--current-mix8` driver now explicitly selects pair0, so future use
cannot silently inherit the new default and mislabel paired results as mix8-only.

Next: independent two-source/ragged H8 main-attention CK contract and full-chain
oracle. Existing production decode M128/192 guards must remain unchanged.
The older H16 BLOCK_K32/64 geometry screen already regressed; no claim that
simple larger K tiles are unexplored. No new main-attention speed result yet.
