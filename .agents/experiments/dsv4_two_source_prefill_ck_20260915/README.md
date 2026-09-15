# Independent two-source prefill CK oracle — not connected to service

Original production CK header and decode M128/192 selectors are untouched.
The experimental derivative retains the license and base snapshot at cb0acfe824;
it adds an extend-bank pointer, encoded bit30 source selection, three split-core
KV-load replacements and one sentinel validity replacement. Unused legacy
launchers retain their192-row cap. A separate checked entry launches H8/D512
M1..65536 with splits1/2 and same-device/contiguous/dtype/workspace checks.
Ragged indptr content is trusted internal metadata, not a validated public API.

The encoder concatenates each row's prefix then extend occurrences. Combined
indptr is the sum of existing indptrs: no D2H count, scan or KV-bank copy.
Duplicate occurrences remain; invalid slots map to -1. Prefix and extend share
one online-softmax tile schedule, unlike production Triton's separate region
loops. Mathematical selection is retained, not a claim of bitwise equality.

## Initial contract evidence

Physical GPU4 only, no service overlap. Twenty analytic cases (ten times
splits1/2) pass exactly, including empty, separate banks, duplicates, invalid
slots, tile boundary and Top512+SWA. Ragged encoded IDs and indptr match CPU
expected mappings. Initial random smoke exited1; its snapshot is `smoke.json`.
The instrumented repeat also exited1 and saved `smoke-repro.failure.pt`.

`smoke-repro.json`: failure at mutation34/splits1 under predeclared
atol.003/rtol.02. The failed element[1,0,237] is CK=-.06103515625 versus
FP32=-.06540885568, absolute error.00437369943; global max error.00653672218.
Graph tests were NOT reached; do not claim1000 graph replays passed.

`diagnosis.json` (exit0): same fixed Q/K/indices/sink, physically concatenate
the two small synthetic KV banks only for comparison against existing CK.
The new kernel is byte-identical to original CK. Production Triton with
positive synthetic OOB IDs sanitized to -1 yields-.0654296875 at the failed
element. This supports inherited probability/tile-schedule rounding, not a new
two-bank addressing bug. It does not diagnose the production service's drift.

Next: explicitly test hi+lo probability from the existing refined CK work, at
the same tolerance. Preserve this failure and do not widen tolerances. Only
after correctness passes measure complete encoder+attention+split-reduction,
including workspace footprint. No new throughput result exists yet.
