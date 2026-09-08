# TP8 C32 native AR and exact-M JIT latency root cause

2026-09-08. Live service2268456,localhost30011,TP8/EP1/no-A2A,original checkpoint.
No speculative CLI/environment. No production changes/restart during diagnosis.
GPUs0..7 ownership checked with amd-smi before testing.

## Workloads and metric boundaries

Six waves x32 requests x256 forced output tokens,stream_interval1,T0,
unique salts,common start barrier,seed20908. Each wave outputs8192 tokens.

1. Legacy `dsv4_tp8_diverse_32_input_ids.json`:mixed chat/science/code,440 input
   tokens total. NOT all-code. Earlier C1/C4 records casually called this corpus
   "code tasks"; actual prompt text must be used, not inferred from script names.
   Decode-window median930.025 tok/s. First4 waves total239.56/238.29/238.85/238.61;
   last2 total893.53/892.48 tok/s. Full6-wave median239.207; do not erase cold
   samples or relabel the last2 as a completed multi-round warm acceptance.
   France first9 exact each wave; only2/32 full sequences cross-wave exact.
2. `dsv4_tp4_code_32_input_ids.json`:32 actual distinct coding tasks,737 total
   input tokens. Six waves full requests239.36–240.87 tok/s; common resident
   decode959.29–960.80 tok/s. All192 requests completed256 tokens,finish=length.
   This corpus has no France sentinel; France was checked in the mixed run.
   Output hashes are retained, not claimed cross-wave exact or semantic oracle.

Raw results and per-request hashes in adjacent JSON; selected manifests:
`/tmp/dsv4_tp8_c32_ar_workload_20260908.json` and
`/tmp/dsv4_tp8_c32_ar_code_workload_20260908.json`.
The reusable source manifests and seed20908 are sufficient to reconstruct them.

## Direct observation of the wait

An additional C32 code wave took33.98288s,decode-window8.46930s (~959.82tok/s).
Read-only /proc wchan,fd,locks and scoped process-tree snapshots during this wave:

- TP2/PID2268744 held the JIT build flock and spawned ninja -> hipcc -> clang.
- Seven other ranks waited in `locks_lock_inode_wait` on the same module.
- Device compile command explicitly targeted amdgcn/gfx90a,wave64 bitcode.
- First pair:gate/up and down,M369. Second pair:M368.369+368=737 true input rows.
- At sampled seconds2–24 compilation/locks were visible;25 onward no compiler.
- This is four SERIAL cold-module builds, not eight redundant concurrent builds.
  The file lock is doing its intended work, not an orphaned/stale lock to remove.

Published `.ninja_log` build durations (end minus first start):

| module | seconds |
|---|---:|
| gate/up M369 |5.886|
| down M369 |6.353|
| gate/up M368 |5.910|
| down M368 |6.275|
| total |24.424|

Using end timestamps from zero instead gives24.427s; the3ms difference is Ninja
start offset, immaterial to the conclusion. This explains most of the25.51s
outside the common decode window. Snapshot data is partial due to tool output
truncation; four independent published Ninja logs corroborate the attribution.

## Why repeated requests still compile

`gfx90a_fp4_expert_gemv.py::_jit_gate_up_grouped` and `_jit_down_grouped` put
the exact `m` in `make_cpp_args`. `m` comes from `xq.shape`,not request count.
The C++ grouped kernels and TensorMatcher specialize the exact M. Prefill
admission cap16 splits32 heterogeneous requests into two groups; arrival/order
differences change the groups' token sums even for the same fixed request set.
Each unseen sum requires another pair of compilations. Disk cache is retained;
there is no evidence of eviction or cache deletion here.

Large BF16-CK selector applies only8192<=M<=36864,so these roughly350–400 row
small-prefill batches use raw FP4 grouped kernels,not the previously benchmarked
large-prefill CK path. M32 decode was captured already and remains fast.

## Warm shape control, without prefix caching

Selected16 actual code requests totaling369 tokens; submitted as a single API
batch twice,max_new_tokens1,unique salts.16/16 responses each,all cached_tokens0:
0.219393s and0.226757s,identical first-token IDs. Therefore normal warm small
prefill plus HTTP is hundreds of milliseconds,not25 seconds.
This is an M369 diagnostic,not a full C32 steady benchmark.

## Correct interpretation and next fix

The latency is incurred INSIDE the prefill forward dispatch on the CPU/JIT side;
it is not25 seconds of useful GPU attention/FFN arithmetic or normal20ms delayer.
AIter's nearby "not found tuned config" log only chooses a torch fallback; it
does not itself prove tuning. The compiler/locks/Ninja evidence identifies our
exact-M FP4 grouped modules as the dominant observed cause.

Prefer runtime-M small-prefill kernels with unchanged math and bounds checks,
or a few correctly masked capacity buckets. Keep existing M1/M32 decode graph
specializations untouched. Warm known buckets before readiness. Do not remove
flock,clear JIT cache,or force every tiny prefill into large BF16-CK merely to
hide this defect. User asked diagnosis; no implementation applied this turn.

Scheduler "input throughput" uses time since the previous prefill log,including
idle/decode time (`metrics_reporter.py`),so it cannot time a single forward.
Both observed full-request and resident-window speeds remain in the record.
