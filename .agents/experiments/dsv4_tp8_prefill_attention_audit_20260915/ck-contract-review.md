# CK reuse review: mixed-sentinel defect reproduced; isolated repair passes

The existing Python H8 selector only admits M128/192; its C++ wrapper and
common launcher also cap tokens at192. Do not remove those production guards
to test large prefill. A separate oracle would need its own checked entry.
The reusable implementation is the repo snapshot
`dsv4_unified_sparse_decode_ck.cuh`, not the absent external CK header initially
looked for. Its two-split workspace is M*2*8*514*4 bytes for H8.

Before any performance screen, inspect the sentinel contract. The core loads
out-of-range/negative KV as zero, while its score/probability masks use the
number of listed entries. The direct reference kernel explicitly skips such
slots. An analytic mixed-negative case with Q=0, valid V=1, sink=0 should
return1/2; counting a negative zero-V slot would instead give1/3.
All-negative/empty rows alone cannot reveal this, since their numerator is0.

`check_ck_sentinels.py` is prepared for physicalGPU4 after the current TP8
service ABBA stops. It tests valid, duplicate, empty, all-negative, and mixed
negative rows through the **existing** H8 module, recording output before
failing any exact comparison. No GPU result exists yet. Normal front-packed
production lists may never contain interior negative entries; this is not
evidence that the running AR/prefill service is affected.

## GPU result and narrow repair (supersedes pending notes)

After the TP8 ABBA completed and amd-smi reported no GPU owners, the existing
module reproduced the defect on physicalGPU4: `[0,-1]` and `[-2,1]` returned
0.333984375 instead of0.5 (max_abs0.166015625). Four other initial analytic
cases passed. The probe intentionally exited1, retaining `ck-sentinels.json`.

An isolated JIT module defines `SGLANG_DSV4_CK_SENTINEL_ORACLE`; normal
production builds do not define it, and production selectors remain unchanged.
It checks each original score's slot before softmax, masks invalid entries,
and avoids exp(-inf- -inf) for entirely invalid tiles. It must read the original
index list, not `kv_slots`, which KV prefetch can already overwrite with the
next tile. Duplicate valid occurrences remain separate contributions.

`ck-sentinels-candidate.json`:12 analytic cases all exact, including out-of-range
positive slots, invalid first/middle tiles, both pipeline transitions and
Top512 mixed indices. `ck-sentinels-stress.json`: same12 exact cases,100 random
Q/K/sink/indices mutations at ragged lengths0/17/33/512, max_abs0.0013812184
versus FP32 reference (atol.002,rtol.01);1000 fixed-input graph replays exact,
and changed-input replay exact versus fresh eager output. Probe processes exited0.

These are synthetic finite-input H8 correctness checks, NOT throughput or
full-model logits evidence. No prefill large-M guard has been widened and no
production fix has been enabled. Normal compact valid lists may not exercise
this issue at all. The next performance oracle still needs valid-list
baseline equivalence, actual large-prefill inputs and complete-chain timing.
