# DSV4 arithmetic balanced-chunk planner rejection (2026-09-02)

## Question

Test whether replacing fixed 2304-token chunks with an equal-sized, 64-row
aligned partition improves TP4/EP1 prefill across prompt lengths.  The candidate
kept the number of chunks fixed and was opt-in; multi-request admission was not
changed.

## Workload and correctness

- Physical GCDs 4--7, original checkpoint weights, TP4/EP1/no-A2A.
- Three real code-review prompts built from different repository sources; no
  repeated filler: `scheduler.py` (4096 tokens), `deepseek_v4.py` (5000), and
  `aiter.py` (8192).
- One warmup and five measured TTFT requests per source, one generated token.
- France returned `The capital of France is Paris.` in both arms.

## A/B medians

| prompt | fixed 2304 | balanced candidate | throughput change |
|---|---:|---:|---:|
| 4096 | 1.96255 s / 2087.08 tok/s | 1.87095 s / 2189.26 tok/s | +4.90% |
| 5000 | 2.28949 s / 2183.89 tok/s | 2.38997 s / 2092.07 tok/s | -4.21% |
| 8192 | 3.73624 s / 2192.58 tok/s | 3.92534 s / 2086.95 tok/s | -4.82% |

The candidate used 2048 for 4096/8192 and 1728 for the leading 5000-token
chunks.  Its cross-length geomean is negative and two important lengths regress
over four percent, so all scheduler/env/test code was removed.

## Conclusion

Chunk cost is not a smooth function of row count.  Expert occupancy, MFMA64
padding/scan count, prefix-dependent sparse attention, and collective size make
arithmetic equalization invalid.  A future planner must use a measured
`T(M, prefix_len, service_state)` table and dynamic programming; do not restore
this equal-partition heuristic.

## MFMA64 ownership finding

The current gate/up MFMA kernel produces one 16-column intermediate tile per
task.  A group32 activation-quant group therefore spans two tasks, and the
persistent grid-stride scheduler does not guarantee those tasks share a CTA.
Directly adding group32 quantization to the existing epilogue is unsafe without
cross-CTA synchronization.  The viable oracle is a 32-column-owned task that
executes two MFMA16 tiles in one CTA and then quantizes locally; it must be
benchmarked as a complete gate/quant/down stage before production wiring.
