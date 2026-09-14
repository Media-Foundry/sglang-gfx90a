# Existing CK H8 mixed-invalid index contract

Independent follow-up to the current prefill attention audit, after the full20
comb service ABBA released all GPUs. PhysicalGPU4 only; no service overlap.

The existing H8 CK split kernel turns invalid slots into zero KV but counts
them in its softmax denominator. Q=0, valid KV=1, sink=0 with indices[0,-1]
returns0.333984375 instead of0.5. Baseline probe exits1 intentionally and saves
results; all-negative/empty rows alone would have missed this defect.

An **experiment-only** compile guard `SGLANG_DSV4_CK_SENTINEL_ORACLE` validates
the original current-tile slot for each score, masks invalid keys, and handles
fully invalid tiles without NaN. Do not use prefetched `kv_slots` to validate
current scores. No production Python selector or default compile flags changed.

12 analytic fixtures exact,100 synthetic ragged Q/K/indices/sink mutations
pass FP32-reference tolerance (max_abs0.0013812184,atol.002,rtol.01),1000 graph
replays exact and changed-input replay equals fresh eager. This is NOT bitwise
FP32-reference equivalence, model correctness, or a performance result.

Evidence:
`.agents/experiments/dsv4_tp8_prefill_attention_audit_20260915/`
`check_ck_sentinels.py`, `ck-contract-review.md`, `ck-sentinels.json`,
`ck-sentinels-candidate.json`, `ck-sentinels-stress.json` (source hashes included).

Do not blame current service drift on this without evidence: ordinary callers
produce compact valid lists. Existing H8 selector remains M128/192 only; this
does not implement large-prefill CK or change AR/DSpark execution.
