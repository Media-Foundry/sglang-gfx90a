# Reconstructed TP8/1M baseline validated after paired-graph fault

Live service PID3507653, native TP8/EP1/no-A2A, original checkpoint, paired
graphs OFF, standalone down-uniform OFF. Parent b93d5b64fd. This is the
reconstructed harness profile, not an exact recovered prior process environment.

Eight ranks captured normally in about12.1s; TP0 logs graph memory0.65GB,
available15.64GB (some ranks15.58GB). Runtime max_total_num_tokens=1048576 and
context_len=1048576. No capacity reduction. This startup control localizes the
previous fault to the added paired-capture flow under this profile, but does
not identify the first erroneous GPU operation.

## Correctness and C1

Controller46881 completed. Separate France C32 prefix32/32 exact.
Three fixed real code tasks, two measured rounds each,256tokens:
all6 full completion ID lists exactly equal the existing runtime-M C1 reference.
Median across six measured samples: **84.09159017665363 tok/s**.
Per-task medians:83.9290626551473 /84.28835110943034 /84.0558982552652.
Two rounds do not support a trimmed estimate or a new performance claim.

## C32 real heterogeneous code workload

Before testing, amd-smi found only the owned service tree (18 distinct PIDs).
Controller33361 completed, six waves ×32requests ×256tokens, native AR.
Selected workload SHA256:
`4d7f83aa48de5df30bf819fe33800bb7f12ecfd4e02ce9215188f62bfd16a424`.

| Wave | HTTP aggregate tok/s | Resident M32 tok/s |
|---|---:|---:|
|0 (warmup)|984.858148|1032.517370|
|1|986.835986|1032.937608|
|2|986.690593|1032.415393|
|3|986.909267|1032.565828|
|4|986.854698|1032.400986|
|5|986.942605|1032.727742|

Drop wave0: warm medians **986.854698 HTTP**, **1032.565828 resident** tok/s.
All192 lengths256, finish=length, and hashes recomputed from saved output IDs
passed. Only10/32 requests are full-output exact across all waves. This is not
a deterministic C32 oracle, and completion/hash integrity is not broad semantic
correctness. France was tested separately, not injected into the code corpus.

## Raw artifacts

- `/tmp/dsv4_tp8_reconstructed_baseline_20260908.json`: startup/control state.
- Same stem `.service.log`, `.france.json`, `.c1.json`, `.c32.json`.
- C32 artifact SHA256:
  `803cb18132111897699e1e4556782c08e9f0bf3c573a69476b07f9fbbf8a2b81`.
- Private launch snapshot is mode0600; never publish its environment.

No new speed feature accepted. Keep this working baseline resident. Next paired
capture investigation must account for reused mutable ForwardBatch/attention
metadata and auxiliary stream state: the old small dual-pool oracle lacks them.
The observed failure occurred in candidate warmup, before successful alternate
capture or runtime arm switching. Do not attribute it to new down kernel math
without an isolated first-fault test.
