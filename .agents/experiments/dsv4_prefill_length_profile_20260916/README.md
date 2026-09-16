# Latest checkpoint: length-dependent cost, not a throughput benchmark

Prepared after32K common-FP32/20 acceptance34fa7b0e9c. Do not launch alongside
the shorter-context regression or other GPU work. The next profile uses the
accepted32K candidate settings for8K/16K/32K, not three different arithmetic
policies. One unscored warmup plus three diagnostic waves per length; original
V4/TP8/nativeAR/1Mpool/32Kchunk, real code inputs and unique cache salts.

Run profile.py --length {8k,16k,32k} serially. Then analyze.py --root <length-dir>
--stop-label P<length>-markers-common --forwards-per-wave {4,8,16}.
Marker count is strict; an unexpected admission pattern requires investigation,
not silently reshaping the frames into presumed waves. All43 layers/all8ranks
must be present. For each forward choose the longest outer-envelope rank and
keep every nested/coarse span from that same rank. Exclude the whole warm wave.
This is not a synchronized global multi-rank critical-path reconstruction.

Report milliseconds per input token alongside seconds per wave:32K contains
four times the tokens of8K, so raw wave duration alone cannot explain throughput
scaling. Nested ranges cannot be added to their enclosing ranges. Client HTTP
diagnostic timing is separate from the streaming formal ABBA score.

Analyzer --check-only reproduced all archived8K owner-profile fields exactly.
test_analysis.py duplicates historical frames as synthetic CPU-only inputs to
check4/8/16-forward grouping, warm exclusion, envelope closure, nested scaling,
and missing-rank rejection. These are not new16K/32K hardware measurements.
