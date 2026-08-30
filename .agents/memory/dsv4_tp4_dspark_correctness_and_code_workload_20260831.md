# DSV4 TP4 DSpark correctness and real-code workload (2026-08-31)

## Scope

- 4 GCDs (`HIP_VISIBLE_DEVICES=0,1,2,3`), TP4/EP1, original checkpoint.
- DSpark checkpoint block size is 5; diagnostic runtime block size was 1.
- Correctness smoke tests use the official chat encoding for “What is the
  capital of France?”. Performance manifests use 32 distinct, concrete coding
  tasks and the official DSV4 thinking encoding.
- `amd-smi process` was captured before every GPU experiment.

## Correctness bugs fixed

1. `DsparkDraftSampler.__call__` invoked the Markov `sample_block` twice and
   overwrote the first result. It now invokes it once and passes the selected
   draft hidden state. Unit coverage checks both anchor modes.
2. A runtime gamma shorter than the checkpoint block size updated the worker
   but left the draft model's `gamma`/`block_size` at 5. The worker now keeps
   model-side confidence reshapes on the resolved runtime gamma.
3. `SGLANG_SIMULATE_ACC_LEN` replaced `correct_len` without reselecting the
   target bonus. A zero-draft oracle therefore emitted the target prediction
   from the old accept row (visible as skipped/alternating France tokens).
   `bonus_for_correct_len` now gathers the target argmax at the overridden row.

Tests:

```text
test_dspark_draft_sampler.py + test_dspark_simulated_accept.py
3 passed
```

After the simulated-bonus fix, a zero-draft France request again emits a
semantically correct Paris answer. The target's long greedy coding trajectory
still tends to repeat; this is also present when no draft token is accepted,
so it must not be attributed to DSpark acceptance.

## Scheduler/admission findings

- The earlier ~1k tok/s number was native AR at BS64, not DSpark at BS32.
- Native BS32 was roughly 623 tok/s in the established checkpoint.
- Real diverse DSpark gamma5 acceptance was much lower than the old repeated
  prompt result (about 1.7 accepted tokens rather than 4.2--4.4).
- Gamma1 is not a throughput solution: a speculative step remained about
  95--100 ms while producing only ~1.1--1.5 tokens on diverse traffic.
- DSpark disables mixed chunked prefill and reserves page-aligned future KV
  slots. With a 12,800-token pool, 32 simultaneous HTTP requests entered in
  waves; aggregate request wall time therefore substantially understated the
  transient resident decode rate.
- Turning scheduler overlap off did not repair diverse-output behavior, so the
  initial FutureMap ABA hypothesis was not established. Length instrumentation
  showed correct logical values (`committed=11`, `prefix_gpu=11`) while 256 was
  only the page-aligned allocation boundary.

## Real coding workload

- Generator: `scripts/rocm/generate_dsv4_code_manifest.py`
- Manifest: `.agents/memory/dsv4_tp4_code_32_input_ids.json`
- 32 unique prompts, 20--28 input tokens, official thinking encoding.
- Domains include Python, Rust, JavaScript, SQL, Bash, C++20, Go, C11
  atomics, HIP/CDNA2, CUDA, Triton, parsers, databases, distributed collectives,
  fuzzing, eBPF, and GPU trace analysis.
- The benchmark stores hashes plus the first 16 completion IDs for compact
  first-divergence evidence. A France-first legacy manifest retains the old
  per-round oracle; a pure-code manifest no longer pretends its first request
  is France.

## Next performance step

Return to checkpoint-native gamma5 and measure the real code manifest. Use
confidence/SPS/STS compact verification only after collecting tables from this
workload. Tune target/draft GEMM shapes from the observed verify-tier histogram,
not from repeated-prompt acceptance.
