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

## Completed layer2 complete-input follow-up

`stable-layer2-prepare` completed warmup/A1/B1/A2 and stopped its owned service
(PID1075741). All64 input echoes and completion-ID/text checks passed.
Across corresponding retained requests, both A1/A2 and A1/B1 have zero
differences in the complete rank0 normalized input/QKV and every rank's
complete uncompressed KV tensors. Sampled Q before/after normalization/RoPE
also agrees on all8 ranks. This closes the upstream-input sampling hole for
the corresponding requests at the entrance to this layer.

Expanded samples show rank0 attention-core differences for case15 already
at **position3** (12 BF16 elements, max0.001953125), and all later sampled
positions except0. Position3 is where the first C4 compression group becomes
available under intended causal semantics. It precedes SWA128 and nontrivial
Top512 boundaries. This points to the C4 attention path for the next audit,
not to query production or different request text.

Important limit: `prepare_full_kv` is a **pre-cache** tensor. This run has not
compared actual packed/quantized cache storage, compressed KV, compressor
projections, or logical/physical attention indices. Therefore it does not
prove a specific compressor or cache-address defect, nor exonerate cache
packing. Same-order complete outputs match13/16; cross-order14/16.

Source inspection identifies another un-stabilized projection:
`Compressor._compute_wkv_gate -> linear_bf16_fp32`; the current AIter BF16
branch calls `tgemm.mm(..., otype=x.dtype).float()`, retaining BF16 rounding
before promotion. The previously added local stability selectors do not
cover this operation. Its row sensitivity must be tested on real layer2
inputs, not assumed from the other projections. The legacy generic row-stable
selector also has a M<=4096 guard, so it does not cover this C16 M32768 case.

Layer2 evidence: `stable-layer2-prepare-evidence.tar.gz`,34 files,3,520,514 bytes,
SHA256 `03491d0878612043103f3c81f41d5b6f8095376ec2dccfbb79dfba6263b35d81`.
The packager validates full-input/KV agreement and actual sampled divergence.
No new production code, default flag or performance claim is part of this
follow-up. All8 GCDs were verified idle after the owned service exited.
