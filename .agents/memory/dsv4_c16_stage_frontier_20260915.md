# TP8 C16 same-input stage-frontier diagnostic (2026-09-15)

## Status

Diagnostic completed, **not a numerical fix or a new speed measurement**. Production
checkpoint remains 7953.956103 input tok/s (C16 x approximately 8K, TP8 native AR,
original V4 weights, 1,048,576 logical KV, 32768 chunk). Keep the persistent drift /
prefill optimization goal active.

Two independent fresh services A/B used identical frozen sources and the accepted
query-owner profile. Both completed 16 exact input echoes, zero prefix hits, one
completion token per code request, and France returned `The capital of France is
**Paris**.` Owned services were stopped; `amd-smi process --json` subsequently
reported no running processes on all eight GCDs.

## Why this probe

Previous sparse checkpoints had matching normalized inputs through layer22 and
different layer24 inputs, with case3 logical C4 rows559..591 first affected. That
did not prove the causal error originated between layers22 and24: deferred MHC
residual/post/comb could already differ, and nine sampled rows missed the entire
visible layer24 change.

The new default-off `SGLANG_DSV4_DEBUG_FIRST_DIV_DIR` probe records the first
eligible large EXTEND forward only, layers20..24, every rank. It records complete
tensor SHA256, BLAKE2b-128 hashes for **every row**, and supplemental numeric rows.
It includes the deferred MHC entry state, attention outputs, FFN inputs, router,
Top-K, shared/local outputs and final FFN output. Parameters have complete hashes.
Model execution is not altered, but synchronous D2H copies strongly perturb
timing; this cannot establish uninstrumented determinism or speed.

## Result

- First forward M32767; same complete input IDs, positions, extend lengths,
  prefix lengths and sample-row identities across A/B for every captured rank/layer.
- 1448 paired stage records. Only **two records differ**, both on layer21/rank5:
  `ffn_routed` and `ffn_partial`.
- Each differs on exactly one whole row: global31948, case3 (fourth request),
  absolute request position7373.
- The relevant FFN input, router logits, Top-K IDs/weights and shared output are
  identical. The local partial differs but the final `ffn_out` is identical; all
  captured downstream states through layer24 are identical as well.
- The changed row is outside the numeric sample. Sample max_abs=0 does NOT bound
  its error. We have no full numeric magnitude for that row yet.
- `ffn_routed` can include shared work on some SBO paths. Do not assert a pure
  expert-stage label without checking the actual hook path. Existing callbacks
  show a separate unchanged `ffn_shared` here, but that alone is not an ownership
  proof for every backend.

The earlier case3/position~2236/layer24 propagation **did not reproduce** in this
pair. Dense synchronization may have changed contention or eliminated an ordering
hazard, or the pair simply did not sample that nondeterministic event. We cannot
distinguish these explanations yet. The local FFN difference is consistent with
the already known CK atomic accumulation issue, but this run alone does not prove
the culprit is atomics rather than another expert-path execution detail.

Do not call the global drift solved. Do not enable expensive fixed-slot stage2 on
this evidence alone. Next bounded step: freeze real layer21 FFN inputs/routing and
actual runtime weight/scale contract, then replay the existing large-M CK path and
compare stage1, stage2 FP32 accumulation, BF16 cast and local/shared addition.
Retain full numeric values for every changed row rather than relying on a preset
position sample. Separately confirm any fix under light probes/full E2E.

## Validation and artifacts

`test/registered/unit/layers/test_dsv4_first_divergence.py`: 2 passed; covers
unsampled-row mutation and inactive scope. Experiment `test_analysis.py`: 2 passed;
covers request/prefix span mapping and corrupt payload fail-closed behavior.
Python compilation and model diff whitespace checks passed. Analyzer verifies
every row-hash/sample file SHA before interpreting it.

Experiment directory: `.agents/experiments/dsv4_c16_stage_frontier_20260915/`.
`run.py`, `sweep.py`, `analyze.py`, `archive.py`, `analysis.json` reproduce the
procedure and conclusion. Evidence archive: 2938 files, 2,876,053 bytes,
SHA256 `65c04cc693780d290d72a4ffd665d9778d954d468ddccdb10758d0ceb19cc215`.
Large row-hash/sample tensors remain local (1840 tensor files listed with size and
SHA256 in `tensor-manifest.json`); all stage JSON plus rank0 metadata are archived.
Do not recursively stage A/B raw directories or the unrelated user pickle.
