# V4.1 long-output SQL functional smoke

Service checkpoint `9bb9acc0fa`; PID137272, same TP8/EP1/native-AR/eager
service,32768pool/context, Engram large tables in RAM, original checkpoint.
No model, kernel, precision, graph or launcher-performance change this round.
The user requested stopping after this run for an emergency job; do NOT
automatically begin further GPU tests after this run is archived.

The user then explicitly requested service shutdown. SIGTERM was sent only
to the verified V4.1 parent137272 at13:36:41 HKT. The service reported zero
remaining requests, exited via its process-tree cleanup at13:36:49, and all
eight schedulers disappeared. `amd-smi process` then showed no processes on
GPU0-7; each GPU used10MB VRAM (65510MB free), and port30101 was closed.
No GPU tests or automatic restart are permitted until the user asks to resume.

## Test and safety scope

Four locally authored realistic SQL tasks, not a published benchmark dataset:
latest successful builds, event sessionization, usage deduplication, and
transitive dependency closure. Each prompt requests an explanation and one
read-only SQLite query. Two rounds of four simultaneous HTTP requests use
fixed input IDs, temperature0, natural EOS, max_new_tokens2048. No ignore-EOS
padding, speculative decoding, or artificial random-token prompts.

Each query is compared to independent Python fixture calculations for16seeds,
including empty inputs, timestamp/id ties, failed/cancelled attempts, NULL
token counts, exact1800-second gaps, cycles, duplicate edges and self-loops.
The model is not given fixture outputs. Raw response IDs, selected/top20
logprobs, SQL, database rows and expected/actual results are retained.
Passing this checker means complete response plus SQL functional agreement;
it does NOT certify the whole English explanation.

A benign `bwrap --unshare-all` preflight failed with
`loopback: Failed RTM_NEWADDR: Operation not permitted`. No model-generated
Python or shell was executed. The alternative runs only SQL in an in-memory
database. An authorizer rejects writes, ATTACH, PRAGMA, reading other tables
and unapproved functions including load_extension. A VM/time progress budget
interrupts runaway recursion.14new CPU tests verify positive references,
negative access cases and interruption; the full reusable suite passes
111tests plus16subtests (13.10s). The forbidden ATTACH test did not create
`/tmp/dsv41-should-not-exist.db`.

## Results

First round:4/4SQL queries pass all64fixture checks. Natural EOS lengths:

| Case | Output tokens | HTTP seconds |
|---|---:|---:|
|dependency_closure|673|152.5983|
|latest_build|705|159.3395|
|usage_dedup|751|168.5510|
|event_sessions|766|171.3362|

Second round is complete and also passes4/4queries and64fixture checks:

| Case | Output tokens | HTTP seconds |
|---|---:|---:|
|dependency_closure|560|128.5189|
|latest_build|641|145.7790|
|usage_dedup|815|180.0483|
|event_sessions|762|170.1665|

Total8/8queries,128/128fixture checks. All eight outputs exceed512generated
tokens and finish naturally. Each of the four cases has DIFFERENT output IDs
between rounds; do not claim dynamic-batch bitwise determinism. All eight
selected/top20 logprob streams are finite/aligned, with valid completion-only
IDs and no prefix-cache hits. No teacher-forced follow-up was run because the
user requested stopping after this run for an emergency job.
All eight raw responses, fixed input IDs and fixture results are retained in
`.agents/experiments/dsv41_sql_functional_20260913/responses.tar.gz`.

HTTP times include prefill, cold shapes,
full explanation and drain, and are NOT resident decode throughput or an ABBA
speed comparison. First admission was1+3requests; second was2+2 according to
the service log. No general determinism claim follows from these trials.

## Manual review: query correctness is not explanation correctness

The first dependency-closure query is correct and uses UNION over node IDs.
Its explanation incorrectly claims that excluding 'app' during recursion
can remove other reachable dependencies in this task. All root neighbors
are already anchor rows; revisiting the root cannot discover a neighbor that
was not already seeded. A path to any other reachable node can be simplified
to avoid revisiting the root. Filtering it during traversal is valid here.

It also overstates that carrying the finite (module,requires) edge pair in a
UNION CTE defeats cycle termination. There are only finitely many such pairs;
unbounded path/depth state would be a different situation. A CPU SQLite
enumeration of all512directed graphs on3nodes (including self-loops) confirmed
identical results for node-UNION, early root exclusion, and edge-pair UNION.
This supports the scope-specific review, not general graph algorithms.
The other three first-round queries and their primary reasoning are coherent
on inspection, but no comprehensive natural-language explanation grade is
claimed. These mistakes alone do not locate a model-loading/kernel bug;
there is no independent full-model output oracle for these prompts.

## Reproduction

```bash
bash scripts/rocm/check_dsv41_unit.sh
amd-smi process --gpu 0
/home/pc/anaconda3/envs/DS/bin/python scripts/rocm/check_dsv41_sql_functional.py \
  --output-dir <new-directory> --concurrency 4 --rounds 2 --max-new-tokens 2048
```

Input generation and fixture oracles are in
`scripts/rocm/dsv41_sql_cases.py`; the checker refuses empty rounds/fixtures.
Do not start another run until the user's emergency job is finished and they
ask to resume.
