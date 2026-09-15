# Producer profile closes; one-stage attention becomes the next candidate

Progress on the active drift/C16 goal. No production kernel or launcher changed
this turn. Accepted production remains7953.956103 input tok/s; opt-in owner-Q
producer formal ABBA is8237.205004 (+3.465% to its control), non-bitwise to the
old BLAS, as documented in `dsv4_c16_owner_producer_perf_20260915.md`.

## Fresh candidate profile

`capture_owner.py --producer` reuses the completed candidate launch/fixture and
freezes current source hashes. Same original V4, TP8/EP1/native AR, original
weights,1M logical KV,32768 chunk. Producer diagnostic equality checks disabled;
only existing asynchronous prefill markers enabled. Full16-request input-ID
echoes/zero prefix hits verified on all four waves (64 echoes).

Service1786216 completed one warmup and three traced waves, produced128
rank-forward frames, and stopped cleanly. Actual producer and paired-MHC hits
checked on all eight ranks. Six CPU profile-label/path tests passed. The trace
driver sends the same16 inputs as a batched /generate request; formal throughput
ABBA used independent streaming clients. These diagnostics do not replace ABBA.

Measured warm client wave durations15.930484/15.930671/15.940205 seconds.
For each forward select the longest rank's full envelope, then retain ALL its
stages. No sum of independent per-stage rank maxima. Mean GPU envelope
**15.84204224 seconds**; event/realtime ratio0.99999651..1.00002205;
maximum rank envelope spread4.30112ms. All43-layer spans close to the envelope.

Mutually exclusive coarse seconds/wave:

| Stage | Seconds |
|---|---:|
| Attention MHC/norm |1.872068|
| Attention entry gap |0.000381|
| Attention preparation |2.284506|
| Main sparse attention |2.760351|
| Attention output projection/collective |1.545196|
| FFN MHC/norm |1.890048|
| MoE/collective |5.374035|

Interlayer/outer intervals account for the remaining envelope in raw analysis.
Nested budgets (already included above, DO NOT add again): routed MoE4.378437s;
MoE output collective0.746924s; both pre-mix1.729131s; both post1.369019s;
weighted norm0.571279s; Sinkhorn0.065434s. Index weights0.028549s,
**query producer0.061600s**, compressor0.117189s, owner chain0.292017s.
Owner chain includes logits/Top-K/pack/AG/reconstruction; adjacent0.000187s
tail marker is not Top-K time.

Compared with prior accepted-owner diagnostic, envelope shrank575.437ms and
query producer shrank565.662ms (98.3% of that diagnostic difference). Other
coarse stages barely changed. This is consistent with the formal ABBA benefit,
not a new randomized component attribution. The prior0.627s query budget is
mostly consumed; no basis to keep counting it as available optimization time.

Profile evidence archive:
`.agents/experiments/dsv4_c16_final_profile_20260915/capture-producer-current-evidence.tar.gz`
3,587,871 bytes, SHA256
`78c272ff0d16795aa25ff2d47d7bfe02b7ed2f9e5a22ac0c4f45de7eace7007e`.

## New bounded attention screen (physical GCD4, services stopped)

Existing two-bank H16/D512/K16/one-wave Triton attention defaults to
**num_stages=2** in this compiler. Previous rejected H8, two-wave and volatile-Q
variants do not test changing only pipeline stages. No stage override has yet
been wired into production. This is compiler scheduling, not a new CK kernel.

Same synthetic M8192 causal C128 and varied C4 fixtures; three ABBA cycles,
five launches per event sample, five Q/K/sink mutations including sentinels.

| Family/candidate | Control ms | Candidate ms | Result |
|---|---:|---:|---|
| C128 stages1 |1.985308|1.384969|30.24% less latency, all five bit-exact|
| C4 stages1 |5.930165|4.209370|29.02% less latency, all five bit-exact|
| C128 explicit stages2 |1.980892|1.984540|same binary as default; no benefit|
| C4 explicit stages2 |5.932917|5.928901|same binary as default; no benefit|
| C128 K32 |1.983660|3.534390|slower, non-bitwise; reject|
| C4 K32 |5.919413|9.417530|slower, non-bitwise; reject|

Stages1 compiler metadata: registers376 vs354, spills0 both, LDS16KiB both.
K32: registers447/LDS32KiB, different softmax grouping and max_abs up to
0.00390625. Register metadata alone is not measured occupancy or proof of the
precise ISA scheduling cause. Do not call30% component latency reduction a
30% service throughput improvement.

## Expanded correctness and fixture correction

Ragged prefix/extend lengths cover0/1/15/16/17/127/128/512, sentinel -1 and
repeated indices (preserved as separate occurrences), empty first row, varying
sink. M=1/17/129/8191/8192/32767/32768/65536 all pass stage1 vs stage2 byte
equality and finite-output checks. Each has three mutations; M129 has100,
plus one exact row permutation and1000 exact graph replays after poisoning
output. Graph and row permutation were NOT tested at every M.

Initial `check.json` is a contiguous fixture: multiplying a sliced tensor
materialized compact storage. Logged strides exposed this (Q4096/512/1,
KV512/1), so it does not support non-contiguous claims. Fixed `check.py` scales
the backing tensor in place BEFORE slicing and asserts Q8192/1024/1,
KV1024/2. Entire suite reran into `check-strided.json`, all exact. Retain both
results, do not overwrite the first or mislabel it.

Large random ragged fixtures also favor stages1, but they are not captured
service KV patterns; small-M event timing includes host launch gaps and is not
a decode optimization claim. Full service numeric checks remain mandatory.

Artifacts: `.agents/experiments/dsv4_c16_attention_pipeline_20260915/` contains
screen.py, check.py, screen.json, check.json, check-strided.json. No model or
attention selection semantics changed. Final amd-smi check: no processes on
any of eight GPUs.

## Next action

Wire default-off stages1 narrowly at original-V4 native TP8 large-prefill
dispatch, without affecting decode/spec/draft/V4.1/CP, then run real C16 ABBA
and quality with producer OFF first. This tests a potentially exact-path
benefit independently of the non-bitwise producer. Preserve K16, both KV
streams, original selection/multiplicity and softmax arithmetic. Verify actual
per-rank hits and matched input echoes. Only after success test stacking with
producer; do not promote based on this standalone screen.
