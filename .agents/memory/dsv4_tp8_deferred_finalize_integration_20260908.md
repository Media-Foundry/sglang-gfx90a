# Native TP8 M32 deferred finalize: default-off service integration

Parent 407c3a63eb. Explicit `DeferredDispatch` adapter carries the request-local
intent into AIterRunnerInput, and the existing grouped-down call returns its
call-owned partial/output object. No global/contextvar storage and no changes to
StandardDispatchOutput tuple arity. The normal FusedMoE Tensor slicing path is not
used for the experimental object. Standard dispatcher, TP8/EP1, no internal result
reduce and no DWDP are asserted. Selector miss fails instead of silently using an
uninitialized output. Production default remains disabled.

Model gate: `SGLANG_DSV4_GFX90A_TP8_M32_DEFERRED_FINALIZE=1`, gfx90a/HIP/AIter,
DSV4, native decode, TP8/EP1/M32, shared present and TP-sharded, no TRTLLM bypass.
Uses the same exhaustive eligibility predicate as the M32 native experiment;
does NOT enable shared-after-TopK. Routed down stays before the original shared
stream join; fused reduction+shared add follows it, then the original TP AR.
C1, prefill and speculative do not hit this selector.

## Verified so far (not a completed ABBA result)

- CPU AST/scope tests pass: down-only argument, default-off, excluded topology/
  modes, unchanged relative join/finalize/AR order.
- Prior full-down component and cross-stream graph correctness are recorded in
  `dsv4_tp8_deferred_finalize_contract_20260908.md`.
- First new service PID3349598 successfully captures all graph tiers. There are
  344 hit logs (43 layers x 8 ranks), not merely an environment-variable claim.
- TP0 graph capture 12.07 s, graph memory 0.65 GB, available 15.64 GB. The earlier
  baseline available-memory record was also 15.64 GB; this is coarse log precision,
  NOT byte-exact proof that allocation liveness costs nothing.
- Actual `max_total_num_tokens=1048576` and `context_len=1048576` retained.
- First candidate C1: six measured 256-token sequences match reference exactly.
- France C32: first-nine-token answer-through-EOS exact 32/32. Forced continuation
  beyond EOS is not a general semantic test and is not claimed as one.
- First real-code C32 round ~983.80 HTTP /1029.62 resident tok/s is only a single
  warmup round; do not use it as an accepted gain over historical measurements.

ABBA controller session 8074 remains running at this checkpoint. State:
`/tmp/dsv4_tp8_deferred_finalize_abba_20260908.json`.
Protocol A(candidate), B(baseline), B, A; each block two C1 rounds and six real
diverse-code C32 waves x256. Same manifest as prior baselines, first wave discarded.
Harness now includes France C32 before timing this candidate. Full C32 sequence
determinism remains a separate concern; do not infer it from France.

Next: inspect controller state/live PID, let ABBA finish, summarize same-run arms,
check both candidate process hit counts/pool logs, and restore default-off baseline
if rejected. A completed controller leaves the last candidate active; do not
mistake the last A for baseline. No new E2E speed checkpoint yet.
