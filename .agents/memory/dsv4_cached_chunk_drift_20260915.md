# Cached 8K chunk: same-revision propagating routed frontier located

2026-09-15. Original V4-Flash TP8/EP1/native AR, original checkpoint,1M pool.
Production C16x8K throughput remains8384.698465 input tok/s; no performance
claim from this slow diagnostic. Global numerical drift is NOT fixed.

## Missing coverage filled

The previous all43-layer audit captured only the first zero-prefix prefill
forward. It found24 local ffn_routed differences, all erased at the FFN output
in that sample. That was not proof of cached-prefix or model determinism.

Added default-off `SGLANG_DSV4_DEBUG_FIRST_DIV_MIN_PREFIX`. Default0 preserves
the old first-forward behavior.8192 waits for the first eligible ordinary
large-prefill batch containing a cached prefix>=8192, without changing math,
KV selection, weights or normal deployment. Capturing one strong batch identity
still prevents accidentally mixing subsequent forwards.

Two fresh services PID2012734 and2019634, one real16384-token code request,
chunk8192 (diagnostic override, NOT production32768), zero external prefix hit,
one generated token. Both captured the SECOND8192 rows, positions8192..16383,
prefix[8192], across43layers and8ranks. Same complete prompt, runtime source
hashes and actual batch metadata. Prompt echoes exact; both France checks pass;
owned processes stopped and allGCDs free afterwards.

## Observed propagation

3440 stage/rank comparisons; metadata mismatches0;3274 unequal records.
All replicated attention-norm/FFN boundaries agree WITHIN each TP run. This is
cross-run numerical drift, not demonstrated TP-rank disagreement.

- Layer0: all recorded stages byte-exact across runs.
- Layer1: attention stages, FFN input, Top-K IDs and weights all byte-exact.
- First local differences: ffn_routed rank0 row1042; rank6 rows2555,3414,4513.
- Layer1 ffn_out: row4513 differs on ALL8ranks. Absolute position12705.
- Layer2 attn_residual/norm: row4513 differs. Subsequent attention and FFN
  stages show many more changed rows; by layer10 all8192 FFN output rows differ.
- Layer42: all8192 final FFN rows differ; final output token is STILL identical:
  token39111, text`Looking`, in both runs.

Do not say3274 independent bugs, or quantify full numerical error from the
sample_max_abs field. Full tensors/rows are hashed; numerical samples miss
row4513, so their zero delta is not an error maximum.

The frontier supports investigating the layer1 routed stage. It does NOT yet
separate CK stage1/atomic stage2/shared addition. Nor does it prove that ALL
later attention differences originate at row4513: the first chunk's KV was not
captured here, and could already carry cross-run differences at later layers.
Only the matched layer1 inputs and its first propagated FFN output are localized.

## Follow-up completed

Default-off CK capture now allows explicit layer/rank/min-prefix, retaining
historical defaults21/5/0. Layer1/rank6 prefix8192 capture and fixed-input replay
reproduced row4513 drift in isolated stage2 atomics. Fixed-slot whole-model
diagnostic has now passed3440/3440 matched stage/rank comparisons. See
`dsv4_cached_fixed_stage2_20260916.md` for scope and the48.3% component cost.
Keep this diagnostic separate from the performance baseline; not promoted.

Artifacts: `.agents/experiments/dsv4_cached_chunk_drift_20260915/` contains
two complete request/launcher/log sets, row hashes and analysis. The evidence
archive retains JSON plus complete input metadata and selected row-hash payloads;
remaining large tensors stay local with SHA references.
