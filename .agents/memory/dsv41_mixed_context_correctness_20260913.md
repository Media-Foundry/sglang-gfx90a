# V4.1 long-decode / short-prefill correctness

Checkpoint: `361766f173`, original V4.1 Flash checkpoint, TP8/EP1/no-A2A,
native AR/eager,32768pool/context,2304prefill chunk, Engram large tables in
host RAM. Same resident service PID137272/tmux `dsv41-long32k-sort-20260913`.
This is correctness coverage, NOT a performance/ABBA comparison.

## First mixed run (complete)

The new `scripts/rocm/check_dsv41_mixed_context.py` launches the existing strict
long-code checker on the frozen22316-token source-review input. It reads only
new service log lines and launches the existing C4 smoke checker after the
long request has entered the >16K candidate-pruning region. No continuous
HTTP health polling, changed model parameters, or softened answer checks.

At13:16:23, remaining long input was3884tokens. The C4 checker was launched
then. Actual admission occurred AFTER the long prefill finished: two log
markers at13:16:32/13:16:35 show four new short sequences with one running
long decode request. Therefore the covered case is long-context decode
mixed with short prefill, NOT two concurrent long-prefill chunks. A missing
overlap marker would make the coverage check fail even if answers were good.

- Long request:22316prompt tokens,90completion IDs, natural EOS, zero prefix
  hit, all7typed JSON keys/values correct. HTTP96.223s (includes prefill).
- Short requests:2rounds x4questions,8/8 pass and natural EOS. France gives
  Paris, arithmetic437, Python`xs[::-1]`, SQL`SELECT COUNT(*) FROM events;`.
- All90long completion IDs match the earlier single-request result from
  the bounded-score-only service. Selected-token logprobs do NOT all match:
  max difference0.10069852; top20 rows also differ. Do not call this numerical
  bitwise parity or attribute the difference solely to batching: comparison
  spans the stable-sort workspace change as well as different admission.
- First mixed run wall102.050s. No OOM or scheduler exception in its retained
  log window. `mixed-context-run1.tar.gz` contains raw requests/responses,
  selected/top20 logprobs for the long request, checker stdout, and log window.

## Second mixed run (complete)

The identical-input repeat also passes: long HTTP94.230s,90completion IDs,
all7typed answers correct, plus8/8short requests correct. Actual overlap
markers at13:20:12/13:20:15 again show4new short sequences and1running decode.
All90long IDs, all90selected-token logprob rows, and all90top20 rows are
identical between the two mixed runs. Every short case also has the same
IDs across its4requests. Thus these two mixed trials pass18/18semantic
requests; this remains limited repeat evidence, not all-batch determinism.

`mixed-context-run2.tar.gz` retains the second raw response/log window.
The service stays running with the same model, context/pool, and weights.
The unrelated31K JSON-key alias failure remains a strict schema failure in
the previous record; these22K passes do not erase it.

## Reproduction and test infrastructure

The fixed prepared input is committed at
`.agents/experiments/dsv41_long_context_20260913/prepared-input.json.gz`.
Decompress to a NEW JSON path, then, with the isolated service running:

```bash
amd-smi process --gpu 0
python scripts/rocm/check_dsv41_mixed_context.py \
  --prepared <prepared.json> \
  --server-log /tmp/dsv41-long32k-sort-20260913.log \
  --output-dir <new-output-directory>
```

Use the DS conda Python. The wrapper invokes the unchanged long-code and
C4 checkers, preserves all their raw reports, and leaves the model service
untouched on client failure. The first CLI invocation rejected a6000pending
default because that was marginally before the16K boundary for22316tokens;
no HTTP request was sent. The corrected default is5000pending.

Seven CPU tests cover log parsing, absent markers, decode-only lines and
prefill-with-running-decode recognition. The new `check_dsv41_unit.sh` sets
the actual local AOT import paths and runs the V4.1 contract suite; this
avoids repeating the missing`sgl_kernel` collection errors recorded in the
workspace note. It passes97tests+16subtests (13.50s). The consistency/startup
gate now runs this suite by default; `DSV41_RUN_UNIT_TESTS=0` explicitly skips
it and prints that fact. The full consistency gate was run successfully after
wiring this default:48complete indexed shards,96085tensors, no missing shard
or tensor, both Engram layers' header/row probes successful, and97tests plus
16subtests pass again (12.46s for the pytest stage). These counts include
checkpoint metadata for MTP/vision; they do not validate MTP/vision inference.
Neither unit markers nor these smoke prompts prove broad
model quality, full-model numerical equivalence, multimodal correctness,
graph support, or1M-context capacity.
