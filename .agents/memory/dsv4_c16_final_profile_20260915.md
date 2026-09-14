# Updated C16 prefill stage budget after query16/runtime-M

## Completed diagnostic, not a new throughput ABBA

`dsv4_c16_final_profile_20260915/capture-v2` completed one warmup plus three
measured diagnostic waves, then stopped cleanly. All eight GPUs were checked
free. Original V4, TP8/EP1/no-A2A/native AR, original checkpoint,16 real8K code
requests (131069 input IDs), zero cache hits, chunk/max-prefill32768 and
1,048,576 logical KV. All64 responses echoed their input IDs exactly.

Actual logs verified both MHC optimizations and query16/runtime-M for all
eight ranks, plus eight observations of the legacy unranked empty-tile print.
Compile-shape logging was disabled for this diagnostic. Full128 marker
snapshots were written;32 warmup snapshots are excluded. No measurement is
spliced from the earlier failed harness run.

For each forward choose the rank with the longest outer GPU envelope and
retain all its component intervals, rather than summing independent maxima.
Cross-rank clocks are not assumed synchronized. Nested spans include metadata,
launch gaps and waits; they are not standalone kernel durations.

Warm client wave times:19.101805/19.106217/19.100895s.
Mean summed GPU forward envelope: **19.010644s/wave**.
GPU-event/realtime ratios1.00004371..1.00006732; maximum rank envelope spread
4.56768ms. The previous detailed diagnostic was23.4889s before MHC and query
reuse; this is a budget update, not a new paired performance measurement.

## Remaining costs (seconds per wave)

| Span | Seconds | Relationship |
|---|---:|---|
| Routed MoE |4.3770|Nested inside MoE/collective5.3677s|
| Two MHC FP32 pre-mix boundaries |3.0602|1.5120 attention +1.5482 FFN|
| Main sparse attention |2.7670|Coarse call envelope|
| Attention output projection/collective |1.5807|Coarse call envelope|
| Indexer logits + metadata |1.4004|Nested in attention preparation|
| Two MHC post-combine/RMS boundaries |1.3629|0.6734 attention +0.6895 FFN|
| MoE output collective |0.7443|Already included in MoE/collective|
| Core compressor |0.6715|Already included in attention preparation|
| Indexer query producer |0.6225|Already included in attention preparation|
| MHC weighted sum/norm |0.5715|Both boundaries|
| QKV projection |0.5633|Already included in attention preparation|
| Indexer Top-K + metadata |0.1366|Nested span|
| Indexer compressor |0.1167|Nested span|
| MHC Sinkhorn |0.0638|Both boundaries|
| Indexer weights projection |0.0279|Nested span|

Do not add this table: parent/child spans overlap. Coarse mutually partitioned
means: attention MHC/norm2.5257s; entry gap0.00038s; preparation4.0950s;
sparse attention2.7670s; output1.5807s; FFN MHC/norm2.5599s;
MoE/collective5.3677s; remaining interlayer/outer gaps close the19.0106s
envelope in `analysis.json`.

## Consequences

Routed MoE remains4.38s, close to the old diagnostic4.37s. Main attention
remains2.77s, close to2.75s. The large measured changes are where expected:
pre-mix4.957->3.060s; post2.813->1.363s; logits2.638->1.400s. Do not count
the old removed costs as available optimization budget again.

Next: bounded MHC pre-mix4->8 reuse screen with all accumulators/masks/stores,
same per-row FP32/K1024 arithmetic, permutation/irregular-M tests and resource
inspection. Require roughly10% stable full-component gain before service
ABBA. Even that would save only about0.306s if fully on this critical path,
about1.6% of the current envelope, not a route to10k by itself. Top-K and
Sinkhorn remain low priorities; long-prefix query coverage is separate work.

## Profile default promotion

After the completed group4->16 warm ABBA and runtime-M serving validation,
the combined TP8 multi-request + prefill-throughput launcher now defaults to
`SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE=16` and
`SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M=1`. Explicit group4/8 and runtime0
are preserved; other profiles remain unchanged. The helper's global fallback
still defaults group4/runtime-off. Model guards exclude native decode,
speculative paths, V4.1, TP4 and unsupported prefill shapes.

The default assignment reproduces the explicitly tested B configuration; no
arithmetic was changed after ABBA/profile. CPU validation:27 tests,47 subtests
passed, including profile conjunctions and overrides; bash syntax and diff
checks passed. One unrelated pytest asyncio_mode warning remains.

## Artifacts

Directory `.agents/experiments/dsv4_c16_final_profile_20260915/`:
`capture-v2/{analysis.json,details-analysis.json,markers/}` contains accepted
data. `capture/` is the failed logger-check warmup, explicitly separate.
Archive214 files,4632427 bytes; SHA256
`37bba4a7233eb9f738166d9208ed80baed99e7f29f5106b1154616b54cf4dcf4`.
The manifest includes every member's SHA256. No model tensors/weights included.

Full-model drift is not declared fixed: the separate runtime-M ABBA retained
the A1-only case4 wording divergence after33 tokens; candidate and A2 were
repeatable and cross-identical. See the runtime service note for exact scope.
