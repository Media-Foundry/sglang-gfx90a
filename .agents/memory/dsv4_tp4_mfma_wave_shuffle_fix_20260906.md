# TP4 MFMA prefill: divergent wave shuffle defect

## Scope

TP4 first; TP8 migration is deferred. Original packed FP4 weights are unchanged.
This supersedes the earlier assumption that the sorter itself was proven faulty.
Repeated requests without unique cache salts also mix full-prefill and cache-hit
execution, so they do not establish a cold-JIT-only problem.

## Concrete defect and controlled test

In `gfx90a_fp4_expert_down_mfma32_kernel`, the final router-weight `__shfl`
executed inside `assignment_valid[half][r]`. Ragged expert blocks can disable
the lane holding the weight needed by another, valid destination lane.
The correction broadcasts before entering that lane-divergent branch.
Accumulation, quantization, expert ordering and reduction order are unchanged.

New regression: `scripts/rocm/check_dsv4_mfma_down_broadcast.py`.
Physical GPU 4, seed 12345; E256/M2304/T6/N64/K512, split2, random unique
top-k IDs and routing weights. Compare broadcast=0 versus broadcast=1 with
identical MFMA math and sorter outputs, for assignments32 and64.

- Before fix: 6/6 comparisons fail (three per assignment size).
- A32: 9,605–10,608 mismatched BF16 outputs; max abs up to 0.03369140625.
- A64: 28,122–29,828 mismatches; max abs up to 0.041168212890625.
- After fix: 40/40 comparisons exact (20 per size), max abs 0.

This establishes a kernel defect, not yet complete model correctness.
Both MFMA selectors remain default-off until end-to-end validation is sufficient.

## TP4 E2E: partial recovery, residual drift remains

Native TP4/EP1/no-A2A, GCD4–7, pool65536, chunk2304, graphBS1,
overlap schedule disabled, both MFMA selectors explicitly enabled.
Service log: `/tmp/dsv4_tp4_wave_shuffle_fix.log`.

- `/tmp/dsv4_tp4_shuffle_fix_c16.json`: 16 distinct code prompts, three
  rounds, 36864 input tokens per round. All 48 cached-token counts are zero.
  First-token vectors are exact across rounds, including cold first round.
  Input tok/s: 1351.12 (cold), 2357.30, 2478.68. This is HTTP concurrency16,
  not a simultaneous GPU M16 decode benchmark: prefill chunks are M2304.
- `/tmp/dsv4_tp4_shuffle_fix_c1_256.json`: three fresh-cache runs,
  full 256-token outputs still differ, first divergence versus round0 at
  zero-based indices 12 and9. Thus first-token stability is NOT full correctness.
- `/tmp/dsv4_tp4_shuffle_logits_{1,2}.json`: fresh salts for each batch member,
  all cached tokens zero. First tokens identical16/16, but top logprobs differ
  for all16 requests (e.g. request4: -0.411602 versus -0.605516).
- France test returns `The capital of France is **Paris**.` and the expected
  nine IDs `[671,6102,294,8760,344,2619,51119,42499,1]`, finish=stop.

Next: fixed-input whole routed-stage replay with fresh sorter outputs, then
first-divergence tracing as needed. Do not call the remaining logprob drift
harmless rounding without an operator-level measurement. TP8 still deferred.

Follow-up isolated replay (`check_dsv4_mfma_sort_replay.py`, GPU0 while the
TP4 service was idle on4–7): both reduced H512/N64 and TP4 H4096/I512/N4096
shapes pass20/20 fixed-input stage replays with fresh sorter calls. xq, xscale,
gate intermediate, intermediate quant/scale, and routed output are all exact.
Logs: `/tmp/dsv4_mfma_sort_replay{,_full}.stdout`. These synthetic inputs do
not establish runtime intermediate equality; capture real first-divergence next.

Harness changes now retain completion IDs/text/cache counts and completion
equality; next-token oracle now isolates prefix caches with per-request UUIDs.
