# Current paired-column C16 prefill profile

The accepted TP8 C16x8K configuration remains7367.6734 input tok/s, original
V4/original weights/1M KV/32K chunk. A fresh diagnostic of this exact kernel
selection completed on HEAD f0d14fbb22 with128 rank-forward frames and64 exact
input echoes, zero cache hits. Service1567196 stopped cleanly; exec3863 exit0.

Warm GPU forward envelope17.726924s; instrumented client waves17.82629945,
17.81990152,17.81415382s. Selected-longest-rank-per-forward methodology and full
closure are retained. This is diagnostic, not a new peak benchmark.

Remaining main budgets: routed MoE4.379s, main sparse attention2.769s, both
pre-mix1.730s, attention output/projection collective1.578s, indexer logits
and metadata1.405s, both post1.361s. Indexer query.628s, weighted norm.571s;
Top-K.137s and Sinkhorn.066s remain small. Details are nested; do not add their
coarse parents again. The pre-pair2.70s pre-mix budget is superseded.

Full methodology/data:
`.agents/experiments/dsv4_c16_final_profile_20260915/current-pair.md` and
`capture-pair-current/{analysis,details-analysis,plan}.json`.
Capture driver validates explicit pair1, query16/runtime-M, mix8, config20=0,
wide0, original model/TP8/EP1/1M and32K. Historical mix8 driver now forces pair0
to avoid inheriting the promoted default. No production arithmetic changed here.

Read-only next-direction audit: main attention really accepts two BF16 KV
sources with separate ragged prefix/extend index streams. Reusing a decode CK
core must preserve both sources and sentinel/duplicate/sink semantics; do not
widen the live M128/192 selector. Historical H16 K32/64 attention enlargement
already regressed (experimental-switches memory around line800), so don't
present that generic scan as new. First build a separate checked ABI and measure
the complete chain; no new attention improvement has been claimed.

Profile archive165 files,3480732 bytes, SHA256
`d1ebec47abcad3c903088375d11f5350e41377a84f2db4a7efc8aaf3d19c91d6`.
Four path-check CPU tests passed; the new driver validated the exact input
manifest and allowed only the recorded launcher-default change from tested B.
