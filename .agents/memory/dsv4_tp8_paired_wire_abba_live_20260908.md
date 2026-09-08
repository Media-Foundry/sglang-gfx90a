# Same-process paired graph ABBA running

Service3527138 successfully started after the numeric wire fix, with native
TP8/EP1, original checkpoint, 1M KV pool and dense/sparse baseline graphs.
Startup controller95448 completed. Benchmark controller32030 is live.

State: `/tmp/dsv4_tp8_paired_same_process_wire_abba_20260908.json`.
Arm order: candidate / control / control / candidate, one PID throughout;
successful completion switches back to baseline arm. Each block runs separate
C32France, three real C1 code tasks with four measured rounds, then six C32 real
code waves. The controller asserts C1 full IDs against the fixed reference.
It never restarts the service or bypasses rejected control commands.

At this note, block0 candidate has passed France32/32 and C1 full reference
IDs12/12 and is running C32. No control block measurement yet, so no gain claim.
Continue polling controller32030 and the same state, not a new benchmark.

Added `summarize_dsv4_paired_graph_abba.py`: accepts only a completed state,
checks all artifact hashes/completion lengths/finish reasons, cross-arm C1 IDs,
common workload fingerprint and native C32 accept fields. Reports separately
C1 per-task trimmed rate/geomeans, warm HTTP and resident M32 rate (wave0
excluded), plus C32 cross-round exact counts. Hash integrity remains distinct
from model semantic correctness. Four synthetic tests pass: known1%delta,
incomplete-state rejection, corrupted-output rejection and changed-workload
rejection. Synthetic tests are not GPU or performance evidence.

After the controller completes, run:

```
/home/pc/anaconda3/envs/DS/bin/python scripts/rocm/summarize_dsv4_paired_graph_abba.py \
  /tmp/dsv4_tp8_paired_same_process_wire_abba_20260908.json \
  --output /tmp/dsv4_tp8_paired_same_process_wire_summary_20260908.json
```

Then inspect all raw artifact results and capacity logs before any promotion.
