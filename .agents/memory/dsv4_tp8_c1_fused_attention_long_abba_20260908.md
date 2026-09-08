# Extended C1 exact attention follow-up

Baseline aeba305d8b, restored uninstrumented TP8 service PID3243459.
The previous independent fused-C1 two-wave candidate passed765/765 cached
decode top20/selected-logprob comparisons, but short ABBA showed only+0.38%
C1 versus about1.16% variation between its two candidate processes.

Follow-up changes no kernel, math, weights, KV allocation or production
default. Same default-off `SGLANG_DSV4_GFX90A_TP8_C1_ATTN_WARPS2` candidate,
same TP8/EP1/native profile, same1,048,576-token pool, same code manifest.
Increase C1 repetitions from2 to6 per block (18 measured requests/block,
72 total), retain C32 six waves/block and fixed-prefix probes. ProtocolABBA;
B/B share a process as before. Startup resource checks use AMD-SMI.

Controller gains `--c1-rounds` (default2) and rejects recorder-instrumented
services. Summarizer validates exact case/rep coverage, retains all C1 samples
and reports both original median estimator and min/max-trimmed mean (with at
least4 samples per case). Replaying the old artifact preserves every original
comparison metric exactly. Python AST checks pass.

State `/tmp/dsv4_tp8_c1_fused_warps2_long_abba_20260908.json` currently running.
Do not promote before complete correctness/performance evidence. Both prior
and follow-up outcomes must be reported; do not keep only the faster run.
Controller leaves candidate active at completion; restore if not accepted.

## Completed: no stable gain, do not enable

All four blocks complete. C1 median estimator regresses0.6015%; trimmed
estimator83.8743 baseline versus83.3689 candidate tok/s (-0.6026%).
C32 warm982.1179 versus983.3610 (+0.1266%), resident1027.6793 versus1028.5629
(+0.0860%). C32 does not hit this selector and provides a noise cross-check,
not evidence of C32 kernel acceleration.

All72 measured C1 full256-token sequences match reference. All24 fixed-prefix
prefill probes have exact IDs/input-logprobs/output-top-logprobs within the
experiment. All768 C32 requests complete256 tokens with hashes validated.
These checks do not establish arbitrary dynamic-batch bitwise parity.

Compare both experiments: initial short ABBA+0.3820% C1, extendedABBA-0.6015%.
Do not select the positive run. Candidate remains default off and performance
retesting is closed absent new evidence; numerical validation from the earlier
765 cached-decode positions remains valid for those tested fixtures.
Adjacent JSON retains every C1 timing and validated summary.

Baseline restoration is running with both two-wave flags removed, same1M KV
pool, under `/tmp/dsv4_tp8_c1_long_restore_20260908`; FranceC32 follows
readiness. The finished candidate service PID3269280 was explicitly verified
and stopped after AMD-SMI ownership checks.

Restoration completed: PID3277900, both attention two-wave flags absent,
FranceC32 32/32 exact. Restore state validated; no instrumented service left.
