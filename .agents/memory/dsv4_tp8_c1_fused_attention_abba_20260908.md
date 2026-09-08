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

Startup evidence: candidate service PID 3198487 logged actual fused C1
two-wave selection on all eight ranks at 17:17:00. No M32 selection log.
Rank 0 reports max_total_num_tokens=1048576 and available_gpu_mem=15.64 GB,
matching baseline capacity. AMD-SMI owner audit found no external GPU PIDs.
All six modified Python files pass AST parsing and scoped diff whitespace
checks. The global dirty-tree whitespace issues belong to unrelated HIP edits
and were not changed. E2E is still pending; this is an experimental checkpoint.
