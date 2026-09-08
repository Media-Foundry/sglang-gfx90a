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

## Layer1 follow-up

Same candidate/config, only dump layer changed; PID2920553.
`/tmp/dsv4_tp8_rowstable_layer1_20260908` contains the tensors.
Layer1 rank0 input, normalized input, Q/core/inverse RoPE, wo_a,
wo_b_partial/global, attention output and FFN output are all736/736 row-exact.
32/32 next-token IDs match reference. No throughput test with debug hooks.
The stage comparison script now accepts `--layer` and `--rank` (default0/0);
its default was rechecked against the layer0 fixture without changing results.

## Next first divergence: layer2 attention/compressor

Layer3 input already differed (664/736 exact, max0.00390625), so the first
learned router is not the first source. Layer2 narrowed it further:
input/norm/Q all736 exact; attention core717 exact, max0.00390625;
global wo_b and FFN output664 exact. Layer2 MHC residual/post/comb all736 exact.
Dump: `/tmp/dsv4_tp8_rowstable_layer2_20260908`, PID2935898 (replaced).

Isolated GPU4 replay of its identical `attn_norm` and raw saved compressor
weights, with USE_AITER=1, BF16_ATTN_LINEAR=1, WAVE64_FP32_GEMV=1:

| Projection | Shape (M,N,K) | Current exact rows | Max row delta | Candidate exact rows |
|---|---|---:|---:|---:|
| core compressor |736,2048,4096|553|0.001953125|736|
| index compressor |736,512,4096|682|0.000244140625|736|

An earlier standalone run without the service AIter flags used another backend;
its ~4e-6 FP32 deltas are not used as the service-path evidence.
The AIter helper returns BF16 GEMM output promoted to FP32. The new diagnostic
hook preserves that BF16-round-then-FP32 contract, scopes through the existing
native TP8 prefill ContextVar, and leaves non-AIter and decode paths unchanged.
It introduces no persistent allocation or weight cache.

## Compressor hook service result

PID2944434, `/tmp/dsv4_tp8_rowstable_compressor_layer2_20260908.log`,
dump directory of the same basename (without `.log`).
Layer2 rank0 input/norm/Q/core/inverse RoPE/wo_a/wo_b_partial/global,
attention output, MHC residual/post/comb, and FFN output now all736/736
identical rows, max delta0, finite. This eliminates the observed layer2
attention discrepancy without changing its indices/cache/addressing kernels.

Three homogeneous32-client waves:96/96 next IDs match the sequential reference.
Final input/output logprobs still not reference-exact; unique arrays per wave
drop12->3, multiplicities8/8/16, multiset identical across all3 rounds.
This is not proof that all remaining error magnitudes shrink or that the whole
model is slot-invariant. France C32 again32/32 passes through EOS.
Raw results `/tmp/dsv4_tp8_rowstable_compressor_probe_20260908.json` and
`/tmp/dsv4_tp8_rowstable_compressor_france_20260908.json`.

Default remains off. Native TP8 decode unchanged; no speed claim, no KV pool
reduction. Next isolate the remaining later-layer or final-logit source before
performance acceptance. Earlier candidate services listed above were terminated
before each replacement; only PID2944434 is current at this checkpoint.

## All-layer first-divergence sweep

Extended the existing dump selector: negative DEBUG_STAGE_LAYER selects all
layers, still gated by debug directory, rank and exact row count. Default
behavior is unchanged. PID2952501 captured all43 rank0 layers, same M736 fixture,
to `/tmp/dsv4_tp8_rowstable_all_layers_20260908`.
All captured stages in layers0..29 are row-exact. First observed difference:
layer30 FFN output728/736 exact, max0.00390625. Layer30 attention and saved MHC
inputs remain exact. By layer42 FFN output540/736 exact, max9.0; do not describe
the entire propagated hidden-state difference as automatically negligible.
Next dump adds `ffn_input` after the fused/nonfused MHC paths converge, to tell
FFN-entry normalization apart from router/expert execution.

Layer30 fused-path FFN input is736/736 exact, max0; output still728/736 exact.
GPU4 isolated router replay using saved [736,4096] BF16 input and [256,4096]
BF16 router weight: current AIter tgemm701/736 row-exact, max0.03125;
fixed-K candidate736/736 exact, max0. The optional prefill scope now also
covers this AIter router wrapper. Return dtype remains BF16. Decode and models
outside the DSV4 scoped forward remain unchanged.
Input dump: `/tmp/dsv4_tp8_rowstable_layer30_20260908`.

## Router hook closes this homogeneous fixture

Current diagnostic PID2968255, log `/tmp/dsv4_tp8_rowstable_router_all_20260908.log`,
all-layer rank0 dump directory of the same basename without `.log`.
All captured stages, including actual FFN input/output, in all43 layers are now
736/736 row-exact and finite on the identical46-token-prefix fixture.
The single wave's final output IDs/input logprobs/output top20 each have just
one unique result across32 clients (previously three logprob variants).

Three additional waves (`/tmp/dsv4_tp8_rowstable_router_replay_20260908.json`):
32/32 clients exactly match themselves across rounds in all three fields.
96/96 next IDs match the sequential reference, but reference logprobs differ:
slot invariance is not bitwise equivalence to the previous library GEMM.
France C32 (`/tmp/dsv4_tp8_rowstable_router_france_20260908.json`):32/32 pass.

This is a fixture-level numerical milestone, not general correctness or a
performance checkpoint. The default is still off, the current service retains
debug dumps, and no new E2E speed is accepted. The original native weights,
1M KV pool and decode paths are retained. Mixed prefixes and debug-free
C1/C32 timing remain separate acceptance requirements.

Mixed six-prefix32-client test completed three waves:
`/tmp/dsv4_tp8_rowstable_router_mixed_20260908.json`.
96/96 next IDs match sequential reference;32/32 clients' IDs, input logprobs
and output top20 are exact across rounds. Reference logprobs remain different,
as expected from the changed projection reduction. This does not cover long
generation, >4096-row fallback, tier transitions or arbitrary prompts.
