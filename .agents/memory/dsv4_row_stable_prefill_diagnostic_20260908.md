# TP8 row-stable prefill diagnostic (in progress)

This is an opt-in diagnostic, not an accepted performance/correctness checkpoint.
`SGLANG_DSV4_GFX90A_ROW_STABLE_PREFILL=1` selects fixed K64-order
BF16 projection only inside native gfx90a TP8/attention-TP8/EP1 extend.
Cached BF16 FP8 linears with contiguous M=5..4096 and N/K divisible by64
are eligible. Decode, speculative verification and TP4 are excluded.
No extra weight cache or persistent workspace is allocated. The second revision
also covers the single-group wo_a fallback. It does not cover compressor helpers,
routed MoE or the final LM head.

Validation so far:

- 128 Boolean selector combinations, default-off scope, nested decode scope,
  and exception cleanup passed.
- Physical GPU4: shapes (M,N,K)=(736,1536,4096), (65,128,256), (5,64,64)
  each passed 100 repeated-row input mutations and 1000 graph replays.
  Empty-M output also checked. FP64 comparison uses tolerances, not bitwise parity.
- Test harness initially compared an unexecuted graph output; corrected to replay
  once before saving the expected tensor. This was not evidence of kernel failure.
- `check_dsv4_row_stable_linear.py` provides the repeatable kernel check.
- Existing standalone real-input fixed-order candidate costs ~113us vs ~90us
  library GEMM; it is a diagnostic numerical tradeoff, not a speed improvement.

Candidate service PID2904693, log
`/tmp/dsv4_tp8_rowstable_diagnostic_20260908.log`, dump directory
`/tmp/dsv4_tp8_slot_rowstable_20260908`. It replaces diagnostic PID2878897
with the same TP8/EP1, 1048576 KV pool, graph tiers1/2/4/8/16/24/32.
Stage-dump overhead means this service must not be used to report production speed.

First service result: 32/32 one-token outputs match the sequential reference;
input/output logprobs are not exact. One wave is not a cross-round stability test.
Layer0 rank0, 16 identical46-token prefixes (736 actual rows):

| Stage | Before exact rows | Candidate exact rows | Candidate max row delta |
|---|---:|---:|---:|
| Q |538|736|0|
| attention core |536|736|0|
| inverse RoPE |536|736|0|
| wo_a |426|574|0.001953125|
| wo_b |260|458|0.03125|
| FFN output |252|482|0.0625|

The first observed discrepancy has moved to the direct wo_a einsum. This is
evidence for a local repair, not a whole-model pass. A second diagnostic revision
adds the same scoped kernel to single-group wo_a fallback. No decode change.
Whole-model and semantic checks remain pending.

## Second revision result

PID2912331, log `/tmp/dsv4_tp8_rowstable_woa_diagnostic_20260908.log`,
dump `/tmp/dsv4_tp8_slot_rowstable_woa_20260908`.
Layer0 rank0: Q, core, inverse RoPE, wo_a, wo_b_partial, wo_b, attention output
and FFN output are now all **736/736 identical rows, max delta0**, all finite.
This closes the observed layer0 slot divergence for this fixture, not every layer.

Three32-client homogeneous waves:96/96 next-token IDs match sequential reference.
Final input/output logprob arrays remain slot-dependent:12 distinct arrays/wave
(previously16), with exactly identical multisets across all3 waves. Per-client
cross-round exact1/32 reflects reassigned batch slots and does not alone imply
a random race. No claim of whole-model logit parity or error magnitude reduction.
France C32:32/32 match the established nine-token answer through EOS.

Raw results:
`/tmp/dsv4_tp8_rowstable_woa_homogeneous_20260908.json`
`/tmp/dsv4_tp8_rowstable_woa_france_c32_20260908.json`.
The flag remains default-off. No new throughput result; do not replace the
accepted C1~82.7 / C32 E2E~980 baseline with diagnostic service timings.
Next: locate the first later-layer divergence with the same fixed-prefix input.
