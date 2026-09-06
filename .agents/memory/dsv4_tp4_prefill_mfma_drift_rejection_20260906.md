# DSV4 TP4 prefill MFMA drift rejection (2026-09-06)

## Scope

This audit used the current `sync/media-foundry-gfx90a` main line, original
DeepSeek-V4-Flash weights, TP4/EP1/no-A2A, a 2304-token chunk, physical GCDs
4--7, greedy one-token continuations, and the fixed heterogeneous code-prompt
manifest `dsv4_prefill_diverse_32_input_ids.json`.  Every GPU launch was
preceded by an `amd-smi process --general --sort-by-pid` resource check.

The purpose was to explain the previously observed first-token drift before
moving the prefill work to TP8.  Scheduler overlap and SBO were disabled for
the isolating runs.  Request-level cache salts were unique in the concurrent
harness.

## Isolation result

The same 2304-token request on a fresh service selected token `43` on the
first MFMA64 execution and token `39111` on the next nine executions.  Turning
off scheduler overlap did not change this sequence.  Precompiling the MFMA64
JIT modules did not fix it either: the first token changed to `32111`, then
returned to `39111`.

Diagnostic initialization of the MFMA gate output and the stage-2 FP32
partial independently did not fix the first execution.  Those diagnostic
changes were removed.

The decisive selector split was:

| Routed prefill path | Fresh C1 result | Heterogeneous C16, 3 rounds |
|---|---|---|
| MFMA64, assignment 64 | cold token differs; warm C1 also produced 3 hashes in 7 rounds | rejected |
| MFMA32, assignment 32 | 5/5 and 7/7 fixed-prompt C1 exact | cross-round request-index drift |
| grouped SDOT, MFMA off | fixed-prompt exact | `cross_round_first_token_exact=true` |

Replacing the gfx90a group-32 quantizer with the reference quantizer did not
remove MFMA32 C16 drift, so the custom input quantizer is not the root cause.
The remaining fault boundary is the MFMA gate/down consumer plus the sorter
contract.  AIter's checked-in CK sorting test matrix uses `unit_size=32`; the
locally introduced `unit_size=64` consumer is not covered by that matrix.

## Drift magnitude

This is not merely bitwise roundoff.  Repeated fixed C16 probes changed
top-1 IDs for the same request index among `39111`, `32111`, `2337`, `671`,
`43`, and `372`.  Examples from three top-5 captures include:

- request 3: `671` at log-prob `-0.3506`, then `39111` at `-0.0193`;
- request 4: `39111` at `-0.7993`, then `43` at `-0.0743`;
- request 12: `2337` at `-0.3952`, then `39111` at `-0.0081`.

Those are semantic trajectory changes, not an acceptable small asynchronous
floating-point delta.  A France smoke test alone is too weak to accept them.

## Performance cost of the safe path

For one real 2304-token source prompt:

- MFMA64 warm median: `2447.00 input tok/s`;
- MFMA32 median: `2334.79 input tok/s`;
- MFMA64 over MFMA32: about `+4.8%`.

For 16 distinct 2304-token prompts (36,864 input tokens total):

- MFMA32/no-overlap median: `2301.97 input tok/s`, not exact;
- grouped SDOT/no-overlap median: `1653.98 input tok/s`, exact.

The large MFMA gain is promising enough to repair, but not safe enough to
ship.  Both `SGLANG_DSV4_GFX90A_FP4_MFMA32_PREFILL` and
`SGLANG_DSV4_GFX90A_FP4_MFMA64_PREFILL` therefore remain available as explicit
experiments while the launch harness defaults both off.  Native decode does
not enter these selectors.

## Artifacts

- `/tmp/dsv4_tp4_mfma32_only_c1.json`
- `/tmp/dsv4_tp4_mfma64_c1.json`
- `/tmp/dsv4_tp4_mfma32_no_overlap_c16.json`
- `/tmp/dsv4_tp4_no_mfma_no_overlap_c16.json`
- `/tmp/dsv4_tp4_mfma32_reference_quant_c16_v2.json`
- `/tmp/dsv4_tp4_mfma32_top5_{1,2,3}.json`

## Next repair step

Capture the sorter outputs and MFMA stage outputs for two distinct prompts in
alternating A/B/B/A order.  Compare every sorted encoded assignment, padded
sentinel, gate/up row, and down partial against the grouped-SDOT consumer.
The first divergent layer and assignment must be identified before either
MFMA selector is re-enabled.  After TP4 C16 is exact, port the corrected
consumer to TP8 and repeat the same heterogeneous manifest and drift audit.
