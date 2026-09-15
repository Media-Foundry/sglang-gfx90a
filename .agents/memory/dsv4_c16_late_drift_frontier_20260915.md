# Same-input late drift: C4 layer22 matches,24 is first observed difference

Accepted owner profile stays7953.9561 input tok/s, original weights/TP8/native
AR/1M KV/32K C16x8K. Two fresh diagnostic services, same16 code inputs and
first-forward M32767 cases0–3, were checked at2 and every even layer20..42.
104 records/service, all8ranks; packed page decode and file hashes revalidated.
Both services passed16 input echoes/zero hits/one output token and France;
both stopped. Diagnostic speed excluded.

Within each service all recorded full x/qlora/Q/weight hashes and logical KV
match across ranks. Across processes2/20/22 match;24 and later differ on all
ranks despite identical input IDs/positions/row placement. This excludes changed
input rows for this observation, not arbitrary other requests/runs.

Important sampling hole: layer24's full hashes differ but all9 sampled rows
have zero numeric differences. Need per-row/full-value capture, not more
confidence in the same small sample. Further, normalized x equality does NOT
establish equality of MHC's deferred multi-stream residual/post/comb state.
Thus22→24 is a visible checkpoint frontier, not a rigorous first-op bracket;
latent differences may be earlier. Do not declare CK atomic the cause yet.

Next diagnostics should include deferred MHC state plus attention and FFN
input/local/reduced output at20–24, then replay the first unequal operator
with proven equal inputs. Historical CK atomic nondeterminism and row-sensitive
projections are distinct established causes, not automatic attribution here.
Fixed-slot oracle historically fits1M KV but was slower; do not enable by
default without current causal/throughput evidence.

Sources/data/analysis:
`.agents/experiments/dsv4_c16_late_drift_20260915/`.
Debug hook now accepts explicit validated even-C4 layer list; existing default
2,20,42 unchanged. Full layer20 fixture can be disabled for this larger audit.
Four CPU capture-contract tests pass. No model/performance selector changed.

Further offline full-cache comparison (kv-rows-v2.json): at layer24 only case3
has changed K/scale rows, exactly contiguous C4 IDs559..591 (33). At layer26
that case changes at559..2047 (1489); other three requests remain unchanged.
This identifies the sampled-query hole and suggests probing raw positions
near2236 with adjacent compressor/SWA context. It is a causal-propagation
hypothesis, not evidence naming the first bad operator. Full dequantized cache
relative-L2 for that request is.000690384 at24,.0240355 at26. Not logits error.
The supplement scripts/JSON are committed separately from the earlier archive.
