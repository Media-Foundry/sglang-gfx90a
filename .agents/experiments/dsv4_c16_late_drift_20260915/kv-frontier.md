# Full saved KV identifies the missed token region

Follow-up after the bounded evidence archive, using its saved rank0 packed
pages and validated logical address decoding. `kv_rows.py` / `kv-rows-v2.json`
compare EVERY valid logical K/scale row, not the9 sampled query rows.

- Layers2/20/22: all valid K/scale rows equal across processes for cases0–3.
- Layer24: cases0–2 remain equal; case3 differs at exactly33 contiguous
  logical C4 rows559..591. Full dequantized-K relative-L2=.000690384,
  max_abs=.0811728 for that request's complete valid cache.
- Layer26: case3 differs at all1489 contiguous rows559..2047; cases0–2
  remain equal. Full dequantized relative-L2=.0240355, max_abs=1.23103.

These are indexed FP8 K values multiplied by their stored FP32 scales; they
are not logits errors or attention-output errors. The records also compare
raw codes/scales, so quantized representation differences are not hidden.

This suggests a local disturbance propagating causally through later
attention. A useful next probe region is raw positions near4*559=2236 in
case3, with neighbors and the preceding deferred residual included. Exact
compressor ownership/overlap must determine the final range; do not equate
that arithmetic hint with the first corrupted raw token. Nor does the33-row
span prove a ring/cache bug or an atomic race. The originating operator and
latent HC-state onset remain open.
