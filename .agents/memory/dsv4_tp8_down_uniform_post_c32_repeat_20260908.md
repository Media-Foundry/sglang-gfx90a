# C1 slow state persists in single-graph candidate

No code, environment, graph or process change. PID3542331 remains the native
TP8/EP1/1M single-graph candidate (DOWN_UNIFORM=1, paired=0). Before measurement,
amd-smi reported18distinct GPU PIDs, all in the owned service tree; no foreign
GPU load. Repeated the same three real C1 code tasks, four measured rounds,
after the completed C32 workload. No freeze-GC request or clock tuning.

Previous C1 trimmed geomean82.7853558; repeat **82.8038589 tok/s**.
All12 measured full completion ID sequences match the fixed C1 reference.
Per-task trimmed rates83 is not implied: actual values82.8556645,
82.8530793,82.7029253. One low sample82.2543 is excluded by the established
per-task min/max trimming, not selectively removed after observing the result.
Adjacent JSON retains all12timings and raw artifact SHA256.

This establishes a stable slow state for this process, not its cause. The
same-process paired experiment kept C1 fixed and found neutral C1 timing, but
that does not isolate differences in JIT module loading or subsequent C1 graph
allocation across newly launched single-graph processes. Do not default-enable
the candidate merely because its C32 kernel is faster. Earlier restored
baseline also had slow C1 state, so the flag alone is not a proven explanation.

Next bounded diagnostic should equalize baseline/candidate JIT module loading
before graph capture, separately from graph-pool geometry. Audit incremental
device memory and retain1M KV. This has not been implemented/tested; do not
claim loading both modules will fix the slowdown. No default or live-service
configuration changed in this repeat; controller10972 completed.
