# Original V4 C16: compressor projection drift reaches actual C4 cache

## Scope and status

TP8/EP1, original checkpoint, native AR, 1M logical KV, M32768/32767.
This is correctness diagnosis, not a performance result. Existing fixed-tile
QKV/wq_b/wo_a/wo_b/shared and FP32 boundary-collective diagnostics were enabled.
They remain default-off and must not be included in production throughput claims.

The `stable-layer2-compressor` fresh service (PID 1090861) completed warmup,
A1/B1/A2 and stopped with no remaining owned processes. All 64 request input
echoes matched the submitted complete IDs; each response's completion count and
decoded output IDs matched its text. B1 exchanges cases 2 and 14, moving case15
by one flat row without changing that request's IDs or positions.

## Actual service evidence

Capture is in the used `compressor_v2.forward_unified` / all-in-one path, not the
old HIP helper. Input/weight/projection are captured before compression; pooled
FP32 values and selected BF16 unified-cache rows are captured after compression
and norm/RoPE/store. Cache rows are compared by (request identity, logical
position), not by physical addresses. The current CompressPlan packs seq_len in
23 low bits and uses a full uint32 ragged_id. The original capture's uint16 mask
was harmless for this <=32768-row run, but is corrected and tested above65535.

A1 versus B1, three common full requests (24576 query rows / 6144 C4 rows):

| Stage | Changed elements | First case15 position | Maximum absolute difference |
|---|---:|---:|---:|
| Complete core/index compressor inputs | 0 | none | 0 |
| Core projection | 10 | 0, column558 | 0.001953125 |
| Index projection | 4 | 256 | 0.00390625 |
| Core pooled FP32 | 8 | 3, column46 | 0.00007422734051942825 |
| Actual compressed BF16 cache | 1 | 3, column46 | 0.00054931640625 |
| Sampled Q | 0 | none | 0 |
| Sampled attention core | 543 | 3 | 0.0078125 |

At query3 the attention core changes12 BF16 elements, max0.001953125.
A1/A2: all these captured stages are exactly equal (32768 query rows,
8192 compressed rows and96 sampled Q/attention rows).

The projection change at source row0/column558 propagates to the first pooled
C4 feature46 and survives BF16 cache storage. This closes the previous gap
between equal uncompressed KV and unknown actual compressed cache. A controlled
projection replacement is still needed before claiming removal of this boundary's
drift. It does not settle later-layer same-order instability.

## Offline replay

`compressor_service_oracle.py` uses captured full inputs and weights, verifies
weights against concatenated raw checkpoint wkv/wgate cast to runtime BF16,
and exactly reproduces all A1/B1/A2 service core/index projections with the
runtime `linear_bf16_fp32` entry. Fixed128/128/128,8-wave BF16-result projection
removes case15 row-placement differences, including ten deterministic input
mutations per projection. It is not bitwise equal to the library output.
See `service-projection-oracle.json` for full comparison counts and source hashes.

Earlier independent timing screen (9-sample median, NOT service ABBA): core
runtime4.675/4.748ms versus fixed5.524/5.506ms; index runtime1.237/1.264ms versus
fixed1.428ms. This is a stability candidate with a component cost, not a speed win.

## Next experiments and reviewer follow-up

First isolate core compressor projection replacement, keeping index projection
unchanged, and recheck actual cache and attention for identical logical inputs.
Do not change cache addresses or Top-K semantics to hide numerical differences.

After this cycle, evaluate original-V4 ordinary-prefill C4 empty-tile skipping
as a separate default-off performance experiment, with valid scores and canonical
Top-K unchanged and C16 service ABBA. Profile the real M32768 critical path;
do not assume indexer dominance or hardware saturation from the 5k plateau.
Then consider same-request multi-query K reuse; large-prefill query-owner
distribution needs a fresh compute-versus-index-exchange cost oracle.

Accepted performance remains the separate 32K/64K records (~5.30k/~5.40k),
not this synchronization-heavy capture run.
