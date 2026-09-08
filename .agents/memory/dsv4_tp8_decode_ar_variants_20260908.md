# TP8 decode AIter AR old/new screen

Base8319c4cb9b; service2727188 unchanged (prep fusion off, M32 overlap on).
Checked amd-smi: no GPU PIDs outside this service tree. Eight-rank isolated
torchrun with HIP_VISIBLE_DEVICES=0..7, numactl --interleave=all, OMP threads1.
No concurrent service requests; no production/library edits or recompiles.

Current service uses AIter custom AR. The available512K/1M block-count hooks
are TP4-only. Do not apply those switches to TP8 and assume they took effect.

New standalone script bench_dsv4_tp8_decode_ar_variants.py compares current
use_new=True (A) with existing use_new=False (B). It uses direct HIP allocation
for registered input, deliberately avoiding caching-allocator IPC offset0 bug.
Communicator max_size1MiB; no model/cache allocations and no persistent changes
to the serving KV pool. Standalone process/context memory is transient.

Five ABBA cycles,200graph replays/sample,20warmups, slowest-rank time per sample.
100independent rank-local input mutations (normal and integers), ten additional
candidate replays per mutation. Every rank: A/B exact100/100, max_abs0, replay
stable. This is relative to existing AR, not a teacher-forced model oracle.

|Shape BF16|Bytes|Current trimmed us|Old trimmed us|Decision|
|---|---:|---:|---:|---|
|[1,4096]|8192|15.10644|15.14905|Neutral/slightly slower; keep current|
|[32,4096]|262144|30.43575|27.88494|Candidate saves2.55081us (8.38% latency)|

No E2E claim. Even86collectives saving2.55us would be only~0.219ms/token-step;
actual applicability and overlap must be measured, not multiplied into a claim.
Next: optional default-off SGLang adapter restricted to TP8/gfx90a native DSV4
decode M32 BF16, preserving C1/prefill/speculative. Benchmark same real32coding
manifest with independent-service ABBA and fixed-prefix/transition correctness.
Do not globally replace AIter or change its default use_new argument.

Raw rank-max samples and all-rank witnesses are in:
/tmp/dsv4_tp8_ar_variants_m32_20260908.log
/tmp/dsv4_tp8_ar_variants_m1_20260908.log
