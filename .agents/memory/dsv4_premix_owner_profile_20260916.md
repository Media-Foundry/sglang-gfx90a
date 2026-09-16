# Updated 9.975k owner profile: pre-mix budget consumed, MoE now one third

2026-09-16, base21e3c6201e. This turn is progress: refreshed complete service
profile and tested/rejected a bounded CK N-stripe speed candidate. Not blocked.

Same exact-owner checkpoint: original V4 Flash TP8/EP1/no-A2A/native AR,
original checkpoint precision,1M logical KV,C16 real diverse8K code inputs,
zero prefix hits,131069tokens/wave,32K admission chunk. All earlier accepted
H16/local-sink/uniqueCK/vec4/exactpost paths stay enabled; MFMA and owner query
producer stay disabled. Formal scoring remains **9975.299479 input tok/s**.

## Closed profile, not a new scoring ABBA

New process, one warmup plus three measured diagnostic waves.128rank/forward
snapshots, all43layer coarse ranges valid. Exclude the entire first wave.
For each forward choose the rank with the largest outer envelope, then retain
ALL stages from that SAME rank. This is not independent stage-rank maxima and
not a globally synchronized multi-rank critical-path reconstruction.

HTTP waves13.16831/13.17877/13.18151seconds. Selected GPU envelope mean
**13.07908seconds**, event/realtime ratio1.0003364..1.0003653. Largest within-
forward rank-envelope spread4.33312ms. Coarse stages plus interlayer/outer gaps
close each envelope. Copies/host recording are asynchronous after each forward.

| Scope | Seconds per wave |
|---|---:|
|Routed MoE|4.357566|
|All MHC/Norm boundaries|1.887170|
|Main sparse attention including current peer path|1.377175|
|Attention output projection + collective|1.538904|
|MoE output collective (nested in larger MoE scope)|0.735553|
|Indexer query producer|0.627513|
|Indexer owner chain|0.290148|

MHC nested detail: post0.825942s, **pre-mix0.399453s**, Sinkhorn0.063193s,
weighted norm0.571456s. These are not additive with the enclosing MHC ranges.

The prior exact-post profile had pre-mix1.766588s; the owner chain now takes
about0.399453s including computation AND exchange/call overhead. Approximately
1.367s of that former budget is already removed. Routed remains essentially
unchanged at4.36s and is now33.3% of the selected envelope. The new profile
supports shifting focus rather than continuing to spend the old1.77s budget.
Steady HTTP overhead remains small; no evidence for a missing multisecond
Python scheduling opportunity in this specific warm waveform.

## Bounded follow-up: CK N-striped partial

See `dsv4_ck_nstripe_20260916.md`: exact FP32 partials and BF16 outputs at six
widths on two real fixtures, but complete stage2 regresses. Keep production
unique Set + fixed vec4 reduction. Smaller scratch is capacity evidence only;
no 1M-pool reduction or service launch with the losing candidate.

## Reproduction

`dsv4_premix_owner_20260916/profile.py`, `profile/analysis.json` and
`profile/details-analysis.json`. Existing analyzer now accepts explicit
`--root` and `--stop-label`, and refuses overwriting prior results; default old
paths are retained. `profile-evidence.tar.gz` and its manifest contain full
raw marker/launch/input/response/lifecycle data, separately from scoring archive.
All sources were hash-checked during the run. Service stopped before isolated
N-stripe GPU tests. No production arithmetic or default selector changed.

Next useful MoE work needs a different supply/writeback organization or current
stage1 hardware evidence. Do not turn the negative stripe result into a claim
that all possible limited-workspace designs are impossible, or extrapolate a
single-GCD stage2 measurement into full routed/service throughput.
