# TP8 M32 delayed shared fork — in-progress experiment

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
