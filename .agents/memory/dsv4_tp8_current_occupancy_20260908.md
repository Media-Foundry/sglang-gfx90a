# Refresh native TP8 C32 real-code occupancy

Baseline 42ae0c5242, service PID3223021 before diagnostic restart. Recent
prefetched-gate screens were synthetic and gave no robust win. The historical
diverse raw dump `/tmp/expert_distribution_recorder_1787803355.1855972.pt`
no longer exists. Refresh real routing before further work decomposition.

Reuse existing SGLang stat recorder, buffer192, not per-token route storage.
Main counter buffer: 192*43*256*4 = 8,454,144 bytes/GCD (about8.06MiB), plus
recorder gatherer overhead. Original TP8/EP1/native settings and explicit
1,048,576-token pool are retained. AMD-SMI audit found only baseline owners
before restarting its verified process tree. No precision/attention changes.

Workload: `/tmp/dsv4_tp8_c32_ar_code_workload_20260908.json`, all32 distinct
code requests, generated168tokens each. FranceC32 is a separate preceding
correctness sentinel. Existing collector now supports explicit
`--skip-france-check` for all-code corpora and preserves all output IDs;
its default still requires the France sentinel as request0.

Analysis drops32 complete warm C32 passes and requires128 complete passes,
separating hash layers0–2 and learned layers3–42. Stat counts are TP-summed,
so use the existing TP8 analyzer division/validation. Logical scan/weight
bytes are not hardware HBM counters.

Job prefix `/tmp/dsv4_tp8_current_occupancy_20260908`: service log/state,
France, collection, analysis JSON/CSV. Currently pending. Recorder adds
instrumentation and its dump calls empty_cache; do not use its request speed
as a new throughput checkpoint. Remove recorder flags and rebuild graphs
before subsequent E2E measurements.

## Completed diagnostic

France32/32 passes; 32 code requests each returned168 tokens. Recorder has
192 buffered passes,168 complete C32 passes. Analyze complete indices34–161,
128 passes after32 warm passes. Eight dumped tensors have identical counts;
all carry rank=0 metadata, so do not claim independent per-rank timing from
these duplicate exports. Every selected layer has192 assignments after TP8
division. Strengthened analyzer to reject negative/noninteger counts and
require each layer's total rather than only the aggregate total. An actual
predicate test rejects missing-layer/duplicate-layer cancellation.

Hash layers: mean active127.04, A4 scans128.34, max occupancy mean4.87.
Learned layers: active82.08, A4 scans95.15, max occupancy mean17.12 (p95=29).
All layers: A4 scans97.46 versus A8 scans88.19, only9.51% theoretical scan
reduction. Layer40 has the largest mean reduction,13.93%; no layer reaches
20%. This closes the contemplated layer-selective A8 direction for this
workload under the earlier scan-saving gate; do not reimplement larger
accumulator ownership merely because some experts have long runs.

Full raw dump `/tmp/expert_distribution_recorder_1788860446.3842516.pt`;
all-code output IDs and hashes in collection JSON. Adjacent memory JSON
retains aggregate and all43 per-layer occupancy metrics. Unlike synthetic
fixtures, these counts include current real code-generation routing, but
histograms alone do not preserve token-to-expert correlations.

Diagnostic throughput includes recorder/dump overhead and is not comparable
to the uninstrumented ~983 tok/s warm baseline. Restoration with recorder
flags removed is running under `/tmp/dsv4_tp8_occupancy_restore_20260908`.

Restoration completed: baseline PID3243459, recorder flags removed, France
C32 32/32 exact. Restore state is validated. No profiling service remains.
