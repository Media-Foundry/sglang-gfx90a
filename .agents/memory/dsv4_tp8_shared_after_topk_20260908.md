# TP8 M32 delayed shared fork — negative service ABBA

Parent ac2d8bf5de. Existing dual-stream path forks shared before router, while
issuing its capture after routed compute. Historical layer20 raw timestamps
show shared starts inside the router interval. That does not prove causality
for the difference between local GEMM and instrumented router span.

Default-off SGLANG_DSV4_GFX90A_TP8_M32_SHARED_AFTER_TOPK moves the existing
alt-stream wait from before router to after TopK, retaining overlap with routed
expert compute. No added events, weight cache, changed tensor math or KV layout.
Guard requires DSV4, HIP/gfx90a, native decode, actual M32, TP8 and EP1. C1,
prefill, other batches/architectures and speculative paths excluded.

CPU AST checks use actual predicate and branch statements: excluded variants
reject; exactly one wait executes on both arms; candidate fork occurs after
mark18/TopK and before expert compute. This is not GPU correctness evidence.

Full service A(candidate)/B/B/A started via run_dsv4_rowstable_abba.py.
State /tmp/dsv4_tp8_shared_after_topk_abba_20260908.json.
Initial candidate PID3301315. Baseline PID3277900 stopped by the validated
harness process-tree lifecycle. Pool1048576 retained. Two C1 rounds/block,
six C32 diverse-code waves/block. No performance or correctness claim yet.
Candidate selection logs identify actual hit per layer. Must inspect full
completion and restore baseline if rejected. Do not enable by default.

## Completed ABBA

All four blocks completed. Both candidate processes logged344 actual selected
layers (43x8). Pool1048576 retained; rank0 reported15.64GB available after
allocation. No new persistent workspace or weight copy.

|Metric|Baseline|Candidate|Change|
|---|---:|---:|---:|
|C1 HTTP tok/s|84.0653|83.8578|-0.2468%|
|C32 warm HTTP aggregate tok/s|984.6539|980.6367|-0.4080%|
|C32 resident tok/s|1030.4566|1026.5605|-0.3781%|

Discard C32 wave0 per block; geometric mean of two block medians per arm.
C1 is excluded by the guard, so its small variation is not a kernel speedup or
proof of causal regression. No confidence interval or independent repeat claim.
Decision: negative on target C32; keep disabled, do not chase this fork point.

24 measured C1 sequences match full256 IDs;24 fixed-prefix prefill probes
match IDs/input-logprobs/top20 across same-run blocks. These probes do not test
cached decode.768 C32 requests validated length256/finish and hashes.
Cross-round exact counts5/6/8/9 of32: neither arm establishes full dynamic-batch
determinism; completion checks are not a semantic or bitwise correctness oracle.

Historical raw profile re-analysis:256 rank samples, all have shared/router
overlap. Per-rank median overlap18.4us, router25.92us, shared112.8us; none had
shared ending after routed. This motivated the trial but is not attribution
of a removable18.4us stall. End-to-end results overrule the hypothesis.

Baseline restoration started from verified candidate3317419, removing only
the experiment flag; exact process tree terminated, no foreign GPU owners.
State /tmp/dsv4_tp8_shared_after_topk_restore_20260908_state.json.
France validation is pending until that job completes; do not claim restored
service ready based on process existence alone.
