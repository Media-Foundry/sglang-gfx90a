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

## Completed follow-up: numerical repair passes, large-M speed rejected

The initial failure and sources are retained in commit1785bad8c8. The optional
`SGLANG_PREFILL_TWO_SOURCE_REFINE_PROB` macro exists only in the experimental
copy; it adds the existing hi+lo BF16 probability decomposition and a second PV
MFMA accumulation. Production sources/launchers remain unchanged.

`smoke-refined.json` exited0: same20 analytic cases exact,100 random mutations
at unchanged atol.003/rtol.02 for both splits1/2, exact index mappings,1000
fixed-input splits2 graph replays and changed-input/ragged-boundary replay
equal to fresh eager output. Maximum absolute error over all mutations is
.0077226162; the test uses combined absolute+relative tolerance, NOT a claim
that every error is <=.003. These finite synthetic inputs are not real layer
captures or full-model logits. `diagnosis-refined.json` fixes the saved element
to-.0654296875, matching Triton, versus FP32-.0654088557.

Complete GPU consumer timings include index encoding, CK core and split
reduction; shared existing ragged-metadata construction and allocation are
excluded for both sides. Each candidate has three ABBA cycles/five calls per
event sample. Contiguous Q is a best-case contract here. Data are synthetic
causal8K requests, not actual model queries or a service benchmark.

| M | family | Triton ms | CK split1 ms | CK split2 ms |
|---:|---|---:|---:|---:|
|8192|C128|1.9900|4.0845|4.6221|
|8192|C4 reused selections|5.9327|12.7002|13.1490|
|8192|C4 varied selections|5.9429|12.6869|13.1389|
|32768|C128|8.2523|16.1391|18.3392|
|32768|C4 reused selections|24.1659|50.3226|52.2916|
|32768|C4 varied selections|24.1610|50.3144|52.2784|

Both benchmark processes exited0, sampled FP32-reference checks passed and
all output elements were finite. Splits2 capacity was allocated for both
comparators:269484032 bytes atM8192,1077936128 atM32768; split1 needs half if
deployed alone. C4 combined indices use18.84/75.36MB respectively. No service
KV-capacity proof is implied. The strided-Q benchmark option is not exercised.

Decision: reject this direct decode-core reuse for production prefill. Both
sizes are roughly2x slower even without Q-copy cost. Do not claim that all
CK-style prefill designs are disproven, or assign the slowdown to a specific
hardware counter without profiling. No E2E run is justified by these results;
the existing7367.67 input tok/s service checkpoint is unchanged. All tests used
only physicalGPU4 after the TP8 service released GPUs; no service remains.
