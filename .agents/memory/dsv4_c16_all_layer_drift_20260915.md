# All43-layer fixed-prefill frontier:24 local differences, no propagated delta

2026-09-15, original V4-Flash TP8/EP1/native AR, original checkpoint,1M logical
KV, chunk32768, query-ownerON, optional producerOFF, accepted attention stages1.
Current default C16x8K performance remains8384.698465 input tok/s; no timed
service optimization in this run. Global model drift is NOT solved.

## Diagnostic coverage extension

Default-off dsv4_first_divergence supports explicit layer and stage selection.
Existing default coverage20..24 is preserved. New env keys:

- SGLANG_DSV4_DEBUG_FIRST_DIV_LAYERS: comma-separated IDs0..42.
- SGLANG_DSV4_DEBUG_FIRST_DIV_STAGES: optional identifier whitelist.

No enable directory means no probe. Existing original-V4 TP8 native-large-prefill
scope remains required. No arithmetic/kernel/weight changes. Selected stages:
entry_prev_residual, attn_residual, attn_norm, attn_core, attn_out, ffn_input,
ffn_topk_ids, ffn_topk_weights, ffn_routed, ffn_out. Every tensor gets a complete
SHA and every token row a128-bit hash; sampled numeric rows are supplemental.

The frontier covers the FIRST prefill forward only. Its input is entirely fixed
prompt tokens, not independently sampled continuations. It is not a complete
cached-decode/teacher-forced logits oracle. Subsequent forwards/request output
processing are not covered by the43-layer state dump.

## Initial diagnostic failure, not accepted

First service1944014 completed its43-layer files but the300s scheduler watchdog
fired during the very slow full-state CPU copies/hashing. Log explicitly reports
watchdog timeouts at21:42:13–14, then SIGQUIT and process-tree kill. The client
received RemoteDisconnected; cleanup raced an already-gone PID and raised
psutil.NoSuchProcess. No complete request/quality acceptance for this A run.
Missing py-spy warnings are auxiliary, not the cause of the timeout.

Keep failed_run.py and A directory intact. Retry generated an isolated launcher
copy adding only --watchdog-timeout1800 to serve; the production launcher and
its default300s protection were NOT edited. Retry uses new directories A2/B2,
asserts actual process arguments, and preserves the same sources between arms.

## Completed same-version A2/B2

Fresh service PIDs1953707 and1962942 both completed and stopped with remaining=[].
Each sent the same16 real code prompts/131069 tokens, one output token, unique
cache salts. All32 complete prompt echoes exact and cache counts zero. Actual
captured first forward is M32767, across all43 layers and all8ranks.

The analyzer validates COMPLETE input_ids, positions, extend/prefix lengths,
row count and metadata across arms, across ranks and across layers. Same runtime
source hashes and same input manifest. No row permutation or prompt/tokenizer
difference is hidden in the following comparison.

| Check | Result |
|---|---|
|Compared stage/rank records|3440|
|Unequal records|24, all ffn_routed|
|Rows changed per unequal record|1–2|
|First local difference|layer3/rank0 row22281; also layer3/rank5 row9938|
|FFN input and Top-K IDs/weights|all compared tensors byte-exact|
|Final FFN output across arms|all43 layers x8 ranks byte-exact|
|Attention norm/core/out across arms|all compared tensors byte-exact|
|Intra-TP replicated norm/FFN boundaries|no differences in either arm|
|Final one-token responses|16/16 equal|
|France|both answer Paris|

The24 local deltas do not survive the subsequent sum/rounding in this run.
ffn_routed can include fused shared output; this probe alone does not separate
CK stage1, stage2 and shared contributions. Prior fixed-input CK replay already
established stage2 atomic nondeterminism, but that must not be used to assert
that every local delta here has been individually replayed and attributed.
No changed row happened to be in the numeric sample set: sample_max_abs=0 is
NOT a full-tensor error maximum and is NOT evidence of zero numerical error.

Heavy probes serialize work and alter rank arrival timing. Absence of a
propagated delta in two diagnostic runs does not prove natural service repeat
stability, cache correctness, batch invariance, or absence of longer-output drift.

## Additional historical clue, strictly separate

Compared the failed A state's layer22/rank0 with the historical stage-frontier A.
Complete metadata/input IDs/positions match. FFN input, IDs and route weights
are byte-exact. Only routed row26811/absolute position2236/column2357 changes:

- Local routed:-0.06005859375 ->-0.059814453125 (delta0.000244140625).
- FFN output:-0.369140625 ->-0.3671875 (delta0.001953125).

Unlike the completed same-version pair, this difference propagates after AR.
It is a useful targeted capture location, NOT a clean causal A/B: source
revisions differ (including CK helper/debug hooks), and new A request failed.
historical.json and compare_historical.py retain hashes and actual numeric rows.

## Next precision work

Do not 'repair' by changing prompts, skipping experts/KV, or blindly enabling the
known52%-slower full fixed-slot stage2. We now have an all-layer first-forward
control that separates erased local noise from propagating state differences.
Next target is later chunk/cache+decode evolution under fixed continuation,
matching actual rows/positions and capturing a propagating FFN row for stage1/
stage2 replay. Revisit position2236 only after same-revision evidence reproduces
it. A cheaper unique-slot CK store/reduction is a research candidate, not yet
implemented or measured; full-model stability will still need its own E2E gate.

Three CPU scope/fingerprint tests pass; syntax and diff whitespace checks pass.
All services stopped; subsequent isolated MHC screen finished; final amd-smi
reports no process on all8GCDs. User memory-usage pickle preserved.

Artifacts: .agents/experiments/dsv4_c16_all_layer_drift_20260915/.
Archive contains raw JSON records, two canonical input metadata files, selected
changed-row tensors, complete request/response/launcher/log evidence and scripts.
All remaining per-row hash tensors are retained locally with hashes referenced
by their JSON records; the bounded git archive is not the entire tensor corpus.
