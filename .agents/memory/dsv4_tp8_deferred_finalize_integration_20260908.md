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

## First complete ABBA (881ce9d213)

Controller 8074 exited successfully. All four blocks complete; companion
`dsv4_tp8_deferred_finalize_abba_20260908.json` retains raw warmup/measured wave
timings, C1 samples, checksums of full output-ID artifacts and validated summary.

| Block | Candidate | C1 tok/s | C32 HTTP warm | C32 resident warm |
|---|---|---:|---:|---:|
| 0 | on | 83.81357 | 984.70278 | 1030.68183 |
| 1 | off | 83.90572 | 983.04307 | 1028.15556 |
| 2 | off | 83.87244 | 983.00266 | 1028.49324 |
| 3 | on | 83.54728 | 983.86455 | 1029.20486 |

Geomean of block medians: HTTP 983.02286 -> 984.28357 (+0.12825%);
resident 1028.32438 -> 1029.94308 (+0.15741%). Both candidate blocks exceed both
baseline blocks, but there is no confidence interval and process drift is visible.
C1 83.88908 -> 83.68032 (-0.24885%); C1 is selector-excluded, so this observation
does not establish a causal C1 kernel regression (nor prove no system effect).

- 24 C1 measured sequences match historical reference; all six fixed-prefix
  prefill probes per block match this run's first block including logprob fields.
- France C32 answer-through-EOS prefix exact 32/32 for all four blocks.
- 768 code requests have validated 256 completion IDs, finish=length and hashes.
  Cross-round exact requests are 8/8/6/9 of32: full C32 determinism is NOT proven.
- Last candidate PID3365803: 344 actual selector hits, TP0 graph 0.65GB and
  available15.64GB, max_total_num_tokens1048576; same coarse logged allocation
  as first candidate and baseline. No loss of configured KV capacity.

Decision: retain default-off; small positive result merits repeat, not promotion
yet. Second ABBA starts from live candidate3365803 with four C1 rounds/block,
same six C32 waves/manifest and same flags. State:
`/tmp/dsv4_tp8_deferred_finalize_repeat_abba_20260908.json`, controller session4134.
It is still running at this record; inspect live state before acting. No restart
on observation timeout. The completed first run's candidate has not yet been
restored because it is the first arm of this explicit repeat.
