# TP8 native C1 fused inverse-RoPE two-wave experiment

Baseline: 69418fe30b. Independent default-off flag
`SGLANG_DSV4_GFX90A_TP8_C1_ATTN_WARPS2`.

The earlier nonfused M32 two-wave experiment was E2E negative and stays off.
This candidate requires native decode, T=1, TP8/attention-TP8/EP1, gfx90a,
BF16 Q, H8/D512, fused inverse RoPE, nonquantized KV, H16/K16 tiles and
multiple splits. It changes only waves from four to two, retaining stages=2,
split count, math and workspace. No persistent weight cache is introduced.

Prior component evidence: `dsv4_tp8_c1_fused_rope_warps2_20260908.json`:
approximately 17.1–17.4 to 15.1–15.4 us, synthetic mutation and graph replay
exact. This is not evidence of E2E improvement.

Current scope test executes the actual selector AST and checks the independent
C1 flag, excluded shapes/dtypes/TP/EP, C32, nonfused attention, prefill,
speculative mode and scope cleanup. Passed after wiring.

ABBA launched using the verified baseline PID 3184375, preserving its
1,048,576-token pool and NUMA interleave policy. Controller checks AMD-SMI
owners before benchmarks and each service transition.

State: `/tmp/dsv4_tp8_c1_fused_warps2_abba_20260908.json`.
Protocol: A/B/B/A; C1 fixed real code requests plus teacher-forced fixtures;
C32 six waves of 32 distinct code requests per block. B/B share a process.
Results pending. Do not promote or claim a speedup before complete E2E and
correctness checks. The controller leaves A active on successful completion;
explicit restoration is required if the candidate fails acceptance.

First A block: C1 case medians 82.9103 / 83.2903 / 83.1096 tok/s;
France passes and all six 256-token sequences match the historical reference.
However, teacher-forced input/output logprobs match in 0/6 fixtures against
`/tmp/dsv4_runtime_m_c1_B_20260908.json`. Maximum compared input logprob
delta is 0.21915; output top-logprob delta is 0.50000, with 47 nonfloat or
structural differences (including top-token membership/order). Do not call
this bitwise exact based on synthetic kernel fixtures or generated text.
Same-run B comparison is still pending and needed to attribute the difference.

Follow-up: first B case medians 83.1919 / 83.2281 / 83.1201 tok/s.
All six A/B fixed-prefix probes match exactly in input IDs, output IDs,
input logprobs and output top-logprobs. The historical-reference discrepancy
is therefore not evidence of this candidate causing numeric drift.

Important coverage correction: the existing fixed-prefix probes request only
one output token, computed during prefill. They do not exercise the modified
cached-decode kernel. Added `check_dsv4_c1_decode_logprobs.py` for a separate,
untimed 256-token continuous-decode probe. It compares output top-20 and
selected logprobs only at positions with identical prefixes, excludes the
initial prefill token from cached-decode counts, and stops comparison after
the first divergent output. Mock HTTP/first-divergence comparison tests pass;
GPU/service validation is pending after ABBA, to avoid contaminating timing.

Shared-scope registered unit tests were updated for both attention flags;
six tests pass, including attention-only not enabling the AR/gate scopes and
all combinations of native/speculative, prefill/decode and M1/2/16/32/64.
Three ABBA blocks completed; final A service is starting. A separate watcher
waits for the live ABBA controller to mark completion before running the
untimed continuous-decode candidate diagnostic, with an AMD-SMI owner audit.

Startup evidence: candidate service PID 3198487 logged actual fused C1
two-wave selection on all eight ranks at 17:17:00. No M32 selection log.
Rank 0 reports max_total_num_tokens=1048576 and available_gpu_mem=15.64 GB,
matching baseline capacity. AMD-SMI owner audit found no external GPU PIDs.
All six modified Python files pass AST parsing and scoped diff whitespace
checks. The global dirty-tree whitespace issues belong to unrelated HIP edits
and were not changed. E2E is still pending; this is an experimental checkpoint.
