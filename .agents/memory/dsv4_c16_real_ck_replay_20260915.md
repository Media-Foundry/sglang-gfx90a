# Real C16 CK fixture identifies one local drift source (2026-09-15)

## Scope and status

Original V4, TP8/EP1/native AR, accepted query-owner C16 profile, original packed
FP4 weights and 1M logical KV. No new performance default. Accepted C16 prefill
remains **7953.956103 input tok/s**. Global output drift is **not** declared solved.

The previous stage-frontier run found one differing local FFN row on layer21,
rank5, global row31948 (case3 absolute position7373), erased by the final FFN
output. This run captures the actual layer21/rank5 MoE and reproduces a stage2
difference on **the same row31948**, with byte-identical full FFN input.

## Capture and an instrumentation failure

New default-off `SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR` scope only admits the
first original-V4 TP8 large EXTEND layer21/rank5 invocation, never AR/decode,
speculation or V4.1. Frozen tensors include inputs/positions, router outputs,
raw FP4 weights and E8M0 scales, actual expanded BF16 B matrices, sorter outputs,
stage1 intermediate, FP32 stage2 accumulation and BF16 output. No weights change.

First service failed the probe completeness assertion: stage1 was recorded but
stage2 was not. Source inspection found AIter `get_2stage_cfgs`-related metadata
caching binds the stage2 callable in a `functools.partial`; its closure held the
empty probe from the first layer that populated the cache. Stage1's wrapper
instead resolves `aiter.ck_moe_stage1_fwd` dynamically. Fix: resolve only the
diagnostic ContextVar **inside stage2 invocation**, not the outer helper call.
No metadata caching, math, kernel selection or production synchronization changed.
This was a diagnostic ownership bug, not a proven production numerical bug.
The failed service exited and its directory/logs remain under `capture/`.

Fresh `capture-v2/` completed: 16 exact input echoes, zero prefix hits, one token
per code request, France correct, then owned shutdown. Complete fixture size
3,317,305,683 bytes. M32767, H4096, I256, Top6, blockM64; shuffled scales and CK
BF16 weights. All raw tensor files remain local, with byte/file hashes in manifest.
The full hidden SHA equals BOTH prior stage-frontier A and B layer21/rank5 inputs.

## Frozen-input replay on physical GCD5 only

Service was stopped first; all eight GCDs reported no running processes. The
replay never overlaps the service and is not an E2E throughput measurement.

Three full helper replays re-run dequant, sorting, stage1 and stage2:

- Expanded weights byte-exact against the service fixture.
- Meaningful sorter prefixes, valid count, sorted weights, complete stage1 and
  stage2 input tensors byte-exact against the fixture and each other.
- FP32 accumulation differs by up to2.9802322e-8 against the service.
- BF16 output: first replay exact to service, next two differ by one byte / one
  value with max_abs1.9073486e-6. This is not a changed prompt or router result.

Stage2-only 30 executions, frozen intermediate/weights/IDs/weights throughout:

- Do not count the first execution's self-comparison as a trial.
- Atomic FP32: **0/29** non-self comparisons bit-exact, max_abs5.9604645e-8.
- Atomic BF16: **24/29** non-self comparisons bit-exact; five differ, at most two
  changed bytes, max_abs3.0517578e-5. Iteration5 differs precisely on row31948.
- Existing fixed-slot stage2: **29/29** non-self BF16 comparisons bit-exact.
- Fixed-slot is NOT identical to the captured atomic output: 4956 changed bytes
  across4553 rows, max_abs0.00390625, relativeL2~1.503e-5. Stability and agreement
  with the previous reduction order are separate claims.

This establishes a genuine frozen-input CK stage2 nondeterminism and connects it
to the observed service-local row. The FP32 atomic expert accumulation is the
identified mechanism; it can cross a BF16 rounding threshold. It does not prove
all previous global drift comes from this mechanism, nor explain why a separate
earlier run propagated a change near absolute2236/layer24.

## Independent selected-row check and cost

CPU FP64 dot/sum oracle uses rows7675,31948,32734 and their18 selected experts.
Captured runtime FP4 plus E8M0 scales expand byte-exactly to the CK BF16 W2 after
undoing its actual shuffle. The CPU check reuses the production scale-unshuffle
helper: it independently checks nibble decoding and the subsequent dot/sum, but
does NOT independently prove checkpoint loading or the scale-unshuffle mapping.
Initial CPU table retained FP4 -0 while
production's INT8-code conversion canonicalizes it to +0; the failed assertion
was67152 sign-zero bytes and **zero numerically unequal elements** for expert24.
The oracle was corrected to that verified zero-sign contract; production unchanged.

Against rounded FP64, atomic selected rows mismatch1,1,1 output elements;
fixed-slot mismatches0,1,0. RelativeL2 versus FP64 is~0.00163--0.00168 for both,
dominated by final BF16 rounding. These three-row tests are not whole-model quality.

Complete stage2 ABBA, five event-timed calls/leg, warmup excluded; allocations,
zero, remap/reduce and BF16 cast included:

- Atomic center **10.727278 ms**.
- Fixed-slot center **16.325639 ms**, **+52.1881% cost**.
- Not an E2E result, not additive with independently timed layer budgets.

Do not enable fixed-slot globally on this evidence: known extra scratch and the
large component cost remain. It is a valid diagnostic/reproducibility control,
not a speed win. Next steps can separate (a) a cheaper stable stage2 writeback
research path and (b) updated post-query-owner service profiling for more C16
throughput. Any production change still requires same-input and full E2E checks.

## Artifacts / validation

Directory `.agents/experiments/dsv4_c16_real_ck_replay_20260915/`:
`capture.py`, `replay.py`, `reference.py`, JSON reports, failed+successful service
logs and bounded selected-row tensors. `archive.json` identifies the evidence
bundle. Large raw fixtures intentionally excluded from git.

Three CPU capture tests pass, including the cached-stage2 ownership AST guard;
combined with the existing first-divergence tests the final regression is5 passed.
Python syntax checked. All eight GCDs were confirmed idle after the final replay.
Only test probes/model hooks were added; all performance selectors retain defaults.
