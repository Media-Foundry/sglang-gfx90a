# Independent pre-mix columns3/4: rejected component screen

Original V4 C16 prefill follow-up after accepted query-owner profile. Production
uses8 token rows and2 independent output columns per CTA; this standalone changes
only the output-column count to3 or4, reusing each activation across more columns.
Each column retains its own `[1,1024]` FP32 reduction and the same16 K chunks;
this is not a merged BLOCK_N tensor reduction or a lower-precision Fn path.

Only physical GCD4, after the profile service stopped. Input is captured layer0
residual/Fn expanded to synthetic occupancy, not fresh full-layer activations.
Ten scaled mutations per shape, EPS1e-6/1e-5/1e-8, row permutation,1000 graph
replays and changed-input graph replay all matched the current2-column FP32
output bits, for M17/8192/32767/32768. Not a proof for all possible FP32 inputs.
Performance is complete pre-mix plus RMS consumption, three ABBA cycles with
five eager calls per event interval; no service speed claim.

| M | Control2 ms | Columns3 ms | Control2 ms | Columns4 ms |
| ---: | ---: | ---: | ---: | ---: |
|17|0.140753|0.168513|0.140305|0.197426|
|8192|1.380859|1.595149|1.368283|1.591245|
|32767|4.980296|6.155793|4.983288|5.653678|
|32768|5.002424|6.362547|5.003385|6.255842|

Registers78→104/128, reported spills0, LDS0. Increased register pressure is a
plausible reason the extra reuse loses, but no occupancy/counter measurement was
made to establish sole causation. Do not infer that no spill means no resource cost.

Both candidates rejected before production integration. No selector or default
changed, no E2E trial warranted. Keep accepted8-row/2-column implementation and
7953.956103 input tok/s checkpoint. Full outputs, checks, per-leg samples and
HSACO/source hashes are in cols3.json and cols4.json. Sources: candidate.py,
screen.py (adapted from the historical paired-column screen, control updated to
the actual accepted2-column implementation).
