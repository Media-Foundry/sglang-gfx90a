# Same-config length profile: longer indexer and main attention explain scaling

Formal non-diagnostic latest accepted speeds:8K10205.686130,16K10023.408672,
32K9605.065395 input tok/s. All originalV4/TP8/EP1/nativeAR/originalweights,
1M logical KV,32Kchunk,C16 public-code prompts,zero prefix hit. Large-prefill
commonFP32/20 enabled, exact post/premix owner/H16/uniqueSet/vec4/directdequant/
wide-indexer owner enabled, queryproducer and MFMApremix disabled.

New separate diagnostic services used exactly the same configuration at each
length. Each one warm +three measured waves,128/256/512 total rank-forward
snapshots,43layers/frame. Entire warm wave excluded; each forward selects the
longest rank envelope and keeps all its stages. This is not a globally clock-
synchronized critical path nor a sum of independent stage rank maxima.
Session43138 exited0, all three owned services stopped. Source hashes verified.

| Scope (microseconds/input token) |8K|16K|32K|
|---|---:|---:|---:|
|Selected full GPU envelope|97.3955|99.3162|103.9816|
|Indexer owner chain (nested)|2.2117|3.7128|6.6760|
|Main sparse attention incl peer|10.5288|11.0666|12.8374|
|Routed MoE (nested)|30.8987|30.8299|30.8026|
|Both MHC/Norm boundaries|14.4064|14.4021|14.4160|
|Indexer query producer (nested)|4.7866|4.7959|4.7900|
|Attention output projection +collective|11.7115|11.6685|11.6333|

GPU wave envelopes12.76553/26.03485/54.51612seconds for131069/262141/524286
input tokens. From8Kto32K per-token envelope rises6.5862us; owner chain rises
4.4643us and mainattention2.3087us. Their sum slightly exceeds total delta,
offset by small reductions elsewhere. Thus measured scaling is primarily
longer-history indexer/attention work, not declining MoE or MHC throughput.
These are diagnostic boundary spans including waits, not kernel counters or
exact causal attribution. Do not substitute reciprocal diagnostic times for
formal throughput or add nested ranges to enclosing ranges.

8K updated routed budget is about4.050s/wave after direct dequant, not old4.358s.
8K MHC premix about0.403s incl owner exchange, not the former1.77s budget.
32K owner chain about3.500s/wave; this longer-history work is still visible,
even though8K owner chain is only0.290s. Further long-context optimization can
target it independently without claiming the same8K benefit.

Artifacts in dsv4_prefill_length_profile_20260916: length-comparison.json,
per-length analysis/details, profile-evidence.tar.gz and per-file digest manifest.
CPU analyzer replay matches historical8K fields and synthetic4/8/16-forward
tests pass including missing-rank rejection. Synthetic tests are not GPU data.

Next bounded candidate: route-major stage2 partial storage. Reuse the exact
existing verified unique-Set CK binary, pack intermediate rows into sorted
assignment order, write contiguous route-owned partials, then fixedTop6 inverse-
mapped reduction. It does NOT remove the~6GiB partial read/write traffic; may
lose on input packing or scattered reducer. Include pack+CK+reduce in oracle.
No runtime integration or speed claim yet; production fixed-slot remains intact.
