# After projection/shared repairs: first remaining compressed-attention drift

Base `b6c28b92fd`, fork main. Original V4-Flash, native TP8/EP1, original
weights, 1M logical KV, C16 distinct8K code requests. This diagnostic enables
the existing default-off QKV/wq_b/wo_a/wo_b/shared fixed-order projections and
FP32 attention/FFN collectives. It is not a production throughput benchmark.

## Completed all-layer sampled run

`stable-all-layers` completed warmup/A1/B1/A2,64 exact full input-ID echoes,
completion-ID/text checks, and stopped its owned service (PID1068045).
Sixteen query positions per request cover0,127,128,511,512,513,639,640,1024,
2047,2048,4095,4096,6144,7424,8191, across all43 layers and8 ranks.
Full KV/query-producer dumps were deliberately disabled to bound IO/storage.
The retained final group is A1/A2 M32768 cases12/13/14/15 versus B1 M32767
cases12/13/2/15. This remains a sampled final-group proof, not full coverage.

`trace_layers.py` now refuses missing required stages and empty comparisons,
and separately reports cross-order and same-order first boundary differences.

### Request-reordering comparison

- Layers0/1 have exact sampled attention and FFN outputs across all ranks.
- Layer1 has a router-logit difference that does not change sampled output.
- Layer2 Q remains exact, but **attention core is the first changed sampled
  operator**. Rank0 changes at15 of16 retained case15 positions (all but0),
  beginning at127; max core difference is0.0078125. Projected attention output
  differs at120/384 rank/query rows, max0.03125.
- Config confirms layer2 is the **first C4 layer** (compress ratios start
  0,0,4,128). Under intended causal lengths, query127 needs fewer than512
  compressed keys, so a
  nontrivial Top-512 cutoff is not a sufficient explanation for that early
  difference. This does not exonerate cache indexing or earlier unobserved KV.
- Later layers amplify the differences. Do not treat large downstream error
  as evidence locating the initial defect there.

### Same-order repeat

The first observed difference is layer15 attention core, not layer0.
Rank0 Q is exact; core differs for case14 at6144/7424/8191, max0.015625,
and the projected attention output max is0.01953125. Layer15 is a C128 layer.
This does not establish that the cause is C128 itself: unsampled upstream
positions and cache state are still unobserved.

## Remaining layer0 router difference is not selected

`router_membership_audit.py` checks the previous dense153-position trace on
all8 ranks. Case15 position3328, expert136 changes from-2.9206276e-5 to
-2.9563904e-5. Selected IDs are238/94/132/174/210/198, and both selected IDs
and weights match. The changed expert is not selected. This explains why
that particular observed logit difference did not alter sampled FFN output;
it is not a learned-router invariance proof or justification to ignore all
router drift.

## Next discriminating experiment

`stable-layer2-prepare` keeps the same numerical profile and request identity
protocol, but captures complete normalized input/QKV on rank0 and full
uncompressed KV on every rank at layer2. This first separates unsampled
upstream input differences from compressor/cache/attention effects. Only
after complete inputs agree should we blame a downstream attention kernel.
Compressed KV and physical/logical index plans may require a further capture.
No new speed result or default promotion follows from this diagnostic.

All-layer evidence: `stable-all-layer-evidence.tar.gz`,34 files,6,094,563 bytes,
SHA256 `421b261ca0adcc2d374de85551eaa2e4d29177c4b8b0f723de74b5f69237b562`.
Complete128-token output matches remain14/16 for both A1/A2 and A1/B1.
Tensor dumps stay local; the archive contains input/output witnesses, logs,
launch configuration, summaries and the router-membership audit.
