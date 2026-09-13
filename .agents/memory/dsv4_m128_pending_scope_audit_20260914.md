# M128 scope and evidence audit (2026-09-14)

Source inspected: 3d73a75923. This is a read-only code/history audit, not a new
GPU experiment or a claim that the M128 candidate works on the merged source.
The requested AR P/D matrix is complete separately. The persistent goal still
names M128, while the user subsequently cancelled DSpark speed measurements.
Clarification was requested: cancel the M128 branch too, or retain its strict
DSpark validation without mixing speculative rates into the AR matrix.
No service was restarted pending that answer. Do not mark the whole persistent
goal complete by silently treating this unresolved item as passed.

## Current source facts

- `gfx90a_fp4_down_consumer_quant_oracle.py` accepts M32/64/96/128 and I256/512.
  This is wrapper support, not proof of production selection or correctness.
- The `use_m32_down_consumer` predicate in `moe_runner/aiter.py` actually covers
  M32 OR M64, with w2 shape(256,4096,128), A4, R2 and LDS-unpack constraints.
  Its historical variable name must not be read as an M32-only contract.
  No M128 down-consumer environment/launcher selector was found in the current
  environ.py or rocm_dsv4_flash.sh.
- The current standalone benchmark reconstructs per-token TopK from recorder
  expert counts; it does not replay the original token-to-expert assignments.
  Weights and activations are synthetic; E8M0 scales are fixed to127.
- Initial output/partial comparisons cover each requested CTA choice. In a
  single invocation, mutation and graph stress choose only `stress_ctas`
  (12 if supplied, otherwise the first choice). Do not claim every CTA receives
  full stress from that invocation. Historical separate invocations may differ.
- The stress loops mutate xq, not the route metadata, weights, scales or route
  weights. The two graphs own separate intermediate/partial/output buffers and
  run serially. They do not establish cross-stream serving buffer ownership.
- The C++ entry validates several tensor contracts but has no TensorMatcher
  check for sorted_ids. This is a validation gap, not proof of a real OOB.

## Historical evidence must be interpreted carefully

`dsv4_tp8_dspark_m128_down_consumer_rejection_20260910.md` records a standalone
691.3->630.1us full-chain result and rejected service trials. Those are older
measurements, not rerun here. Its cross-round hash rejection is not by itself a
valid numerical diagnosis: the later `.summary.md` explicitly supersedes the
earlier attribution and records batch/admission-dependent drift in controls.
The old mixed-prompt repetition observations still warrant investigation, but
cannot be assigned to this kernel without a matched control and first-divergence
evidence. Do not call a stream race or a mathematical bug established.

If the user retains this branch, the needed experiment is a scoped strict-M128
revisit using real inputs/routes/scales and explicit workspace/stream ownership,
then matched service correctness and resident-window comparison. Merely rerunning
the synthetic component or substituting an AR M128 toy test would not close that
requirement. No such implementation or GPU validation occurred in this audit.
