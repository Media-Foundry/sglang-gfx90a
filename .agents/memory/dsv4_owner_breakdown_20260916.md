# Real-service owner breakdown: 32K logits is the next component target

Base ef3d40aa32, diagnostic-only marker patch8170d3adce. Slots55..61 in the
existing64-slot per-layer matrix were verified unused in archived real frames.
Import-time disabled unless SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR is set. No
new host synchronization or D2H. CPU AST tests show removing the seven marks
leaves every owner function/class body identical to base; grouping/planning,
scores,TopK,RCCL and reconstruction arithmetic were not changed.

New8K and32K realC16 code services use latest commonFP32/20 checkpoint,TP8,
originalweights/nativeAR,1Mpool,32Kchunk. All64 first-token replies per length
match the same prior profile inputs and outputs exactly, zero prefix hits.
One warm +three scored-as-diagnostic waves,128/512 rank-forward frames,43layers
and21 owner-bearing layers/frame. Same longest outer-envelope rank selected
per forward; six substages plus entry/exit gaps exactly close the owner range.
All ranks/frames including warmup checked. This is diagnostic boundary time,
not standalone kernel throughput or global synchronized critical-path proof.

| Owner chain ms/wave |8K|32K|
|---|---:|---:|
|planning|1.2753|3.8723|
|pack|6.1937|29.2498|
|logits|177.3474|2875.9698|
|TopK|18.3586|186.7472|
|AllGather incl arrival wait|76.0455|344.1807|
|reconstruction|14.3246|61.7090|
|outside detailed interval|0.3771|1.5170|
|full owner chain|293.9221|3503.2457|

32K logits82.09%,TopK5.33%,gather9.82%;8K logits60.34%,gather25.87%,TopK6.25%.
Do not prioritize a new TopK sort algorithm based on the combined owner span.
Planning is tiny and no evidence here of a large missing host synchronization
fix. Ownership already uses round-robin original query16 groups, not contiguous
query ranges; do not repeat round-robin as a new proposal.

Session47613 exited0, both owned services stopped. Source hashes and cleanup
verified in archive.py. profile-evidence.tar.gz contains702 files,17084245bytes;
archive-manifest.json records every digest. Additional synthetic CPU analyzer
tests preserve closure and reject a missing internal mark, even in warmup.

Follow-up isolated scheduling screen in dsv4_indexer_grid_order_20260916:
only change CTA order to process several K tiles near the same query groups,
using the exact existing load_k/emit_runtime_m math. Group sizes1/4/8/16/32
all byte-exact on tested scores/logical+physical IDs and100 graph replays.
SingleGCD GQ1 logits gain~4%32K/~6%8K/~11%mixed-prefix,64registers/0spill.
This is synthetic component evidence, not real service speed or a cache-counter
measurement. No hardware-cache-hit attribution has been proved yet.

Then limited BS16/32/64 screen with GQ1, forcing16x16 MFMA for wider tiles:
BS32 component speedup1.64..1.76x on tested fixtures,92..95regs/0spill and
scores/IDs exact. BS64 scores differ; mixed-prefix mutation0 also changes
logical/physical IDs. BS64 rejected, not passed off as harmless rounding.
K32 high-dynamic-range/non-page-aligned-width stress is a separate next gate;
no production K32 selector or E2E gain yet.8K logits is only~0.177s/wave,32K
~2.876s/wave, so component speedups cannot be applied to entire service time.

Formal speeds remain8K10205.686 /16K10023.409 /32K9605.065 input tok/s.
