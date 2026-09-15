# CK stage2 atomic drift localized; fixed-slot cached-chunk remedy verified

Original V4-Flash TP8/EP1/native AR, original weights,1M logical KV. Performance
default is unchanged at8384.698465 C16x8K input tok/s. No new fixed-slot service
speed is claimed; all-model/global batch-independent determinism remains open.

## Same-input attribution

Following the all43-layer cached-chunk frontier, fresh capture PID2028810
targeted layer1/rank6, M8192 with prefix8192 of the same16384-token code request.
The COMPLETE hidden, Top-K IDs and weights SHA match BOTH preceding A/B captures.
This is not merely a matching row number or synthetic reproduction.

After owned shutdown, only physicalGCD6 ran frozen-input replay:

- Three full dequant/sort/stage1/stage2 repetitions: stage1 intermediate,
  stage2 input, live sorted IDs/weights/expert IDs/counts are all byte-exact
  against service; expanded weights are also checked. Only stage2 FP32
  accumulation and BF16 output vary.
- Thirty isolated atomic stage2 runs:27/29 comparisons to first run have BF16
  drift. All29 later FP32 accumulators differ.
- Exact propagated row4513 varies in iterations9,13,24, at column1951:
  0.0167236328125 versus0.0166015625 (one BF16 ULP, delta0.0001220703125).
- Existing fixed-slot stage2:30 runs, all byte-exact to the first.

Loaded CK module hash:
`9a8d60c45857823a5582bd3401db3ffc1cb70210d62a3a781b8a68ca967e66ac`.
Stage2 blockM64, TopK6, activation1, quant0, use_nt=True. Invocation/kernel
metadata and manifest preserved under the experiment. Capture selectors were
extended default-off, defaults21/5/0 retained. No existing CK binary modified.

## Independent selected-row reference and actual cost

Rows2555,3414,4513,18 selected experts. Decode raw FP4/E8M0 W2 on CPU,
validate the selected BF16 weights byte-for-byte, then FP64 dot/weighted sum on
the frozen BF16 stage2 intermediate. This is a stage2 reference, NOT a full
model FP64 oracle or an oracle for the earlier stage1 nonlinear activation.

Fixed output agrees with BF16-rounded FP64 in all4096 columns of row2555 and
row4513, with one differing column on row3414. Thus repeatability does not mean
every output equals a correctly rounded full-FP64 sum. Max absolute errors are
about1.22e-4,2.07e-4,2.36e-4 respectively, dominated by final BF16 rounding.

Complete stage2 ABBA on physicalGCD6 (allocation/zero/remap/CK/reduce/cast as
applicable, same frozen inputs): atomic3.029114ms; fixed4.491952ms, **+48.29%
cost**. Fixed scratch M8192x6x4096 FP32=768MiB, versus128MiB atomic. This is a
real correctness remedy with a significant component penalty, not a speed win.

## Whole-model controlled intervention

Two new services PID2035828/2042450 enabled ONLY the existing fixed-slot
numerical option relative to the cached-chunk diagnostic setup. Both retained
the1M pool and chunk8192 diagnostic override. Captured second chunk positions
8192..16383, all43layers/all8ranks/the same10 stages. Complete source hashes,
input IDs, positions and actual metadata match between fixed arms.

| Metric | Atomic A/B | Fixed-slot A/B |
|---|---:|---:|
| Stage/rank comparisons |3440|3440|
| Metadata mismatches |0|0|
| Unequal stage/rank tensors |3274|**0**|
| Intra-TP replicated boundary mismatches |0|0|
| Final one-token response |39111 / Looking in both|39111 / Looking in both|

Both prompt echoes/cache checks and France pass; both services stopped with no
remaining owned descendants. This establishes a fix for the observed
same-shape, same-prefix cached-EXTEND drift. Probe placement/timing is the same
within each pair; no speed inference from slow instrumented service times.

Limits: first chunk KV values not independently archived; cached single-token
decode not traced; LM-head/full logits not part of3440; only one code request,
one chunk shape, two processes per arm. Different batch compositions can still
select different GEMM paths. Do NOT label arbitrary free-running responses
bitwise stable, or enable fixed-slot universally based only on this test.

## Next work

Keep fixed-slot as the deterministic diagnostic reference, not default. A
cheaper unique-slot store plus fixed-order reduction is worth measuring, but
must retain all experts, FP32 reduction and BF16 boundaries. The H16 peer
attention component winner is orthogonal; do not add micro savings/penalties
and call them a measured combined service result.

Artifacts:
- `.agents/experiments/dsv4_cached_chunk_drift_20260915/` atomic frontier.
- `.agents/experiments/dsv4_cached_ck_drift_20260915/` exact matching fixture,
  complete replay JSON, selected-row reference and stage2 ABBA.
- `.agents/experiments/dsv4_cached_fixed_drift_20260916/` fixed intervention.

Bounded archives include complete raw JSON/logs/manifests and selected tensor
payloads. Full CK weights/intermediate and remaining row-hash files remain local
with SHA references.11 focused CPU tests passed (preexisting pytest config
warning only); runtime scopes preserve native decode/draft paths.
