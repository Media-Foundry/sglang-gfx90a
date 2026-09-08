# TP8 C1 shared gate rounding fusion: wiring and initial screen

Baseline commit: `3f21c29ca1`. This is a default-off experiment, not an accepted
throughput checkpoint. Original weights, TP8/EP1/no-A2A, native AR, 1048576-token
pool, graph tiers 1/2/4/8/16/24/32. No new persistent weight cache.

## Dispatch fix

The first service screen did not enter the candidate: `DeepseekV2MLP.tp_size`
stores the optional constructor override, normally None. Its column projection
resolves actual TP size. Check `gate_up_proj.tp_size == 8`, not the optional
override. This does not change the constructor or existing paths.

The model-scoped ContextVar enables only gfx90a, native decode, BS1, attention
TP8, global TP8, EP1. The MLP additionally requires cached BF16 gate weights,
no pre-quantized input, contiguous BF16 x[1,4096]/w[512,4096], same GPU, limit10.
All other cases retain existing forward. Nested disabled scopes clear active
state and restore it on exit, including exceptions.

The R1/U2/W4 kernel preserves projection BF16 rounding and intermediate SiLU
BF16 rounding, as tested in the standalone oracle committed in `28a0b44347`.
This is not the prior ordinary GEMV geometry-only 0.309-us candidate.

## Tests and initial results

- CPU `check_dsv4_shared_gate_scope.py`: 256 predicate combinations, nested
  disabled/prefill/spec/C32 scopes, exception restoration, CPU rejection, real
  MLP forward entry executed with optional TP=None and resolved projection TP8.
- All eight ranks logged selection during BS1 graph capture in
  `/tmp/dsv4_tp8_shared_gate_wired_20260908_service.log`.
- Graph memory 0.65GB/GCD, rank0 available15.64GB; runtime pool1048576 unchanged.
- C1 three code cases, two measured repetitions each: case-median geometric
  mean84.1016 tok/s. France exact; all6 measured256-token sequences match the
  existing reference. All6 fixed-prefix output IDs, input logprobs and output
  top-logprobs match the previous unhit screen.
- C32 three waves, 32 distinct code requests, 256tokens each: median E2E982.8546,
  resident1028.6726 tok/s. All96 finish length with256 tokens. Cross-round exact
  requests13/32, so dynamic-batch bitwise determinism remains unresolved.
- The earlier unhit screen was83.5316 C1 tok/s; it is NOT a controlled A/B.
  Historical formal baseline was about84.3. No speedup is claimed from these
  initial numbers.

Raw results:
`/tmp/dsv4_tp8_shared_gate_wired_20260908_{c1,c32}.json`.

## Formal acceptance underway

`run_dsv4_rowstable_abba.py` now takes an explicit candidate flag, leaving the
row-stable-prefill flag fixed when testing shared gate. Protocol A/B/B/A, two
measured C1 repetitions per case plus warmup, six diverse C32 waves per block.
For this candidate it requires all measured C1 output IDs to match reference
before proceeding to C32. It checks AMD-SMI ownership before changing services.

State: `/tmp/dsv4_tp8_shared_gate_abba_20260908.json`.
Inspect live controller/service state before acting; do not restart on an
observation timeout. Results remain pending until all four blocks finish and
warmup-discarded summaries, correctness and memory are inspected.
