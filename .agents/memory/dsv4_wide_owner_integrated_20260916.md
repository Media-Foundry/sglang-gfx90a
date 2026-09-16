# Wide-owner integration: C16x16K +21.5177%, exact measured outputs

Base `6812bb7994`: accepted **8K** C16 prefill 10205.900816 input tok/s.
New trial is **16K**, 262141 actual prompt tokens/wave. Never mix these rates.

Default-off `SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE=1` enables W2049..8192
under existing original-V4/TP8/eager/native/EP1 guards and global M8192..65536.
The compact wrapper now receives admitted global M separately, bounds local
rows by ceil(global_M/128)*16 and requires query16 alignment. No kernel math,
query production, compressor/cache update or Top-K semantics change. W>8192
still falls back. Existing narrow wrapper body is AST-regression checked.

Directory `.agents/experiments/dsv4_wide_owner_integrated_20260916/`.
Eight-rank production-helper oracle finished (not just direct kernel calls):

| fixture | full chain ms | owner chain ms |
|---|---:|---:|
|8K M32768 W2048|18.669177|3.178444|
|16K M32768 W4096|39.014678|5.624609|
|32K ragged M32767 W8192|79.755160|10.401093|
|mixed prefix M32768 W8192|75.272802|10.072892|

Eight-rank slowest sample medians, ABBA x3, includes GPU packing, logits,
Top-K, integer RCCL gather, reconstruction; excludes full query producer and
host planning. Three mutations including cutoff ties, independently permuted
physical page IDs: full scores byte-exact and logical/physical IDs exact.
Do not report these component ratios as E2E speedups.

16 CPU tests pass (owner/scope/capture/producer/direct-dequant), one existing
pytest asyncio_mode warning. `git diff --check` clean.

Real model diagnostic `check/complete.json`: **1344** live comparisons across
all eight ranks passed score-byte/logical-ID/physical-ID checks. France passed,
1,048,576 logical KV/32768 chunk confirmed. Diagnostic rate7253.476 is NOT a
formal score (it repeats full computation and synchronizes). Owned check
service2474308 stopped cleanly.

`run.py` completed owned serial fresh A1 -> B(B1/B2) -> fresh A2, then analyzer.
Each scored leg: three waves; each process: four128-token quality waves plus
fixed64-token continuation based on A1 quality0. Both arms use accepted exact
dequant/premix-owner/HIP-post/H16/unique-Set-vec4 and wide-query16 fallback;
producer off. Only wide-owner flag differs. Source hashes frozen across arms.

## Formal service result

| leg | warm median input tok/s |
|---|---:|
| A1 full wide-query16 |8244.398350|
| B1 wide owner |10025.443262|
| B2 wide owner |10024.105319|
| A2 fresh full wide-query16 |8254.879125|

Control center **8249.638738**, candidate **10024.774291**, **+21.517737%**.
Control A2/A1 drift+0.127126%. All six candidate waves above10019.86.
Raw client times independently rechecked: sum262141 input tokens divided by
max(first-token-time)-min(request-start). All prompt echoes correct, zero cache
hits,16 completed one-token outputs. These are P-wave rates, not decode rates.
Per-process first warmup excluded identically: A1=7529.13, B=8989.84,
A2=7548.70. This is a warm-service comparison, not a cold-TTFT claim.

Three processes x4 quality waves x16 requests x128 output tokens: **192 answers**
and24576 tokens all identical, within and across configurations. All input
echoes, IDs, completion lengths and tokenizer decoding checked. The16 unique
128-token excerpts were read: coherent code-review topics, no obvious garbling
or repetition collapse. They are truncated beginnings, not validated complete
solutions; generated claims of races/bugs/deadlocks were not independently
certified. No universal accuracy or arbitrary-batch determinism claim.

Fixed64-token continuation from A1 quality0: both A1/A2 and A1/B compare
**1008** non-null positions, max/mean selected-token logprob difference0,
1008/1008 Top1 matches and1008/1008 complete Top5 records identical. Exclude
one leading null per request, do not count empty arrays as equality.

Runtime source hashes frozen across diagnostic and all three timed processes.
All four owned services stopped cleanly; final amd-smi showed no processes on
any of8GCDs. 16 CPU tests re-run after services, pass; output inunit.log.
Evidence: summary.json, acceptance.json, service-evidence.tar.gz with per-file
hashes inarchive-manifest.json. `validated-launcher.sh` is the full measured
candidate launcher (not an outer export subsequently overridden by old flags).

No global default promotion. 8K accepted speed remains10205.900816 from the
separate earlier trial; no new8K E2E speed measured here. Narrow8K production
helper oracle and AST body regression passed. **32K/mixed-prefix have only the
eight-rank component proof**, not real-model service acceptance in this trial.
Next: close that coverage independently, including batch1 MHC dispatch and
long-output numerical checks; do not assume16K proves all contexts. Runtime
pickle and unrelated untracked files preserved. Persistent goal remains active.
