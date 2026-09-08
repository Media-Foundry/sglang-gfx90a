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
