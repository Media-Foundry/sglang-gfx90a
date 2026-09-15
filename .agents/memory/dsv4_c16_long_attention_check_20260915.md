# Long-input same-live-input attention check: all16512 calls byte-exact

Active drift/C16 goal progressed. Original V4-Flash, TP8/EP1/native AR,
original weights,1M logical KV,32768 chunk, wide-queryON, producerOFF,
stages1ON. Diagnostic only; no new speed result. Accepted default8K production
remains8384.698465 input tok/s; optional producer stack remains8692.442097.

## Why this test

The preceding uninstrumented16K/32K coverage had repeated-output matches14/16
and7/16 with identical request IDs/cache counts. That does not locate the
first numerical difference. This run tests the suspected attention schedule
at identical LIVE Q/K/indices/sink, before cache publication, instead of
comparing unrelated greedy trajectories.

Existing debug selector computes stages1 and original default-stages2 for
each admitted call and compares output bytes. Added only a debug-branch pass
log containing rank/layer/compression ratio/actual M. Stripping this one print
restores the entire prior backend AST exactly (one CPU test). No arithmetic,
selection, kernel arguments or default-off diagnostic behavior was changed.

## Complete v2 evidence

Fresh service1887482. Two C16 quality waves at16K, then two at32K,128 output
tokens each. Same frozen real code manifests as the preceding service; all64
full input echoes exact, no prefix-cache hits, lengths/tokenizer decoding
checked. No timing waves or throughput claims under this diagnostic.

| Length/wave | Forwards per rank | Exact attention calls across8ranks |
|---|---:|---:|
|16K/0|8|2752|
|16K/1|8|2752|
|32K/0|16|5504|
|32K/1|16|5504|

Total **16512**, not including the earlier incomplete harness run. Every
rank and every layer0..42 covered for every forward; actual M32767/32768.
Ratios match config per layer, including first two ratio0 SWA-only layers,
then4/128. Config's extra entries after43 are not counted as target layers.
All comparisons byte-exact; no mismatched attention output was seen.

Full-response repeats still differ:

| Length | Current diagnostic repeat | Prior uninstrumented repeat | Cross-run match range |
|---|---:|---:|---:|
|16K|13/16|14/16|13..14/16|
|32K|9/16|7/16|5..11/16|

The reference kernel and torch.equal introduce synchronization/extra work,
so these repeat counts are not a clean speed or improved-determinism A/B.
They demonstrate that local attention equality can coexist with full-model
drift. Do not claim that improved32K repeat count is a fix or regression.

## Interpretation

On these observed inputs, replacing stages2 with stages1 does not introduce
attention output differences. This narrows the search, but does NOT establish
identical Q/K/indices between separate waves, cache correctness, equivalence
to an external model oracle, or global determinism. If attention inputs
already differ upstream, both implementations may faithfully produce the
same output for that differing input. Existing FP32-atomic CK stage2 drift
and other upstream/arrival effects remain independent candidates.

The known real CK replay proved a small stage2 difference with fixed input,
while its fixed-slot substitute cost52% more; that rejected implementation
should not silently be enabled to make this diagnostic pass. Long32K still
uses the preexisting batch1 FP16 MHC tail, unlike16K multi-request FP32 mix.
No claim that either explains every observed output change is made here.

## Retained harness error

Initial driver expected ratios{4,128}, omitting legitimate ratio0. It completed
the first16K request wave and2752 exact attention comparisons, then failed its
own coverage assertion. The service stopped through owned cleanup. When I
attempted to interrupt the driver after noticing the bad expectation, psutil
reported it had already exited; no signal was sent. This was not a GPU or
model failure and not a permission issue.

Original failed_run.py and service/ evidence preserved. V2 uses actual config
ratios for the43 target layers and verifies equal per-layer call counts.
V2 results live in service-v2/; it also stopped cleanly. Final amd-smi found no
process on all8GCDs. User pickle/unrelated files preserved.

## Next work

Keep stages1 accepted; no reason from this test to roll back its5.31% gain.
For precision, fixed-continuation first-divergence work should inspect inputs
to MoE/attention rather than treating final hashes as a unique cause. For
performance, a separate wide query-owner oracle is justified by current
W>2048 fallback to replicated scoring, but must preserve full Q/compressor,
all logical selection IDs and exact local scores before an8-rank stage test.

Artifacts `.agents/experiments/dsv4_c16_long_attention_check_20260915/`: driver,
failed driver, per-layer coverage analyzer, observation-only AST test, raw
input/output/check logs and hashed evidence archive. No semantic accuracy
certificate for generated code diagnoses is claimed. Goal remains active.
