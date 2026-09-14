# Original V4 C16: runtime-M query16 serving validation

## Outcome

Fresh TP8/EP1/no-A2A native-AR services completed A1/B1/B2/A2 and stopped
cleanly. Runtime-M retains query16 throughput while removing the extra
compile variant for changing query-row counts. This is a cold-shape latency
fix, not a new arithmetic throughput win. Original checkpoint, full causal
selection and 1,048,576 logical KV tokens remain intact.

| Leg | runtime-M | Three input tok/s waves | Median |
|---|---:|---|---:|
| A1 | 0 | 4593.35 / 6878.88 / 6873.63 | 6873.63 |
| B1 | 1 | 6871.00 / 6882.20 / 6875.51 | 6875.51 |
| B2 | 1 | 6882.23 / 6880.95 / 6862.14 | 6880.95 |
| A2 | 0 | 6879.03 / 6871.50 / 6701.63 | 6871.50 |

Mean leg medians: control6872.5618, candidate6878.2304 input tok/s,
**+0.08248%: effectively equivalent**. All arms are group16, not group4.
The earlier independent group4-to16 ABBA was +1.4533% and remains the
evidence for group16's steady-performance gain.

## Do not hide the cold round

A1's first timed wave encountered a new M32765 compilation. It is retained
in the three-wave record, not silently filtered or relabeled warm. The other
two A1 waves were about6.87k. The analyzer records serving compilation events
for each leg and sets `all_timed_legs_compile_warning_free=false`.

The reported mean leg median request-TTFT13.3177->12.0194s includes this cold
effect. **Do not claim a9.75% warm-TTFT gain.** Excluded initial warmups were
2676.06/6115.97/6126.53 input tok/s for A1/B/A2 respectively. The candidate
binary had already been built by the integrated component test; these are not
matched empty-disk-cache startup measurements.

## Actual compiler evidence, every rank

A1: M32765/32766/32767/32768 with matching width2048, pages32, stride32,
group16 and pointer alignment produced four distinct compiled hashes/rank.
B: observed M32766/32767/32768 reused **one hash and one cached object/rank**.
No serving compilation warning occurred anywhere in B's log. A2 reused
already built constant-M disk-cache variants, also without such warnings.

Candidate hash:
`c1f3f7d98823fee34df57cd7b8fb9f7cadc780bcfc6aa34e8c467f4abf915aa9`.
This matches the integrated component result. It does not imply different
widths/layouts/strides never need an initial compilation.

## Correctness and drift

- Identical source hashes and input manifests in all services;131069 input
  IDs/wave,16 distinct real code tasks, zero prefix-cache hits.
- Same timed forward counts: four rounded-M32768 forwards/wave.
- All eight ranks logged group16 and the expected runtime flag; all four
  preceding optimization families hit. Actual server info retained1M KV.
- France returned Paris in all three fresh services; this short request did
  not activate the large-prefill selector.
- All96 quality responses echoed their supplied input IDs exactly; decoded
  completion IDs matched response text and had128 tokens.
- B and A2 each repeated16/16 and all B/A2 wave pairings were16/16 exact.
  A1 second wave also matched every B/A2 wave16/16.
- A1 first wave alone differed on case4 after33 tokens (979 vs14114).
  The two inspected excerpts are coherent weight-loader concurrency analyses,
  with different organization/wording. Referenced symbols occur in the input.
  This is not evidence that all remaining model drift is fixed.

The integrated helper test passed700 score-bit and logical/physical Top-K
mutations, including odd M and ragged/mixed pages, plus100 fixed graph replays
per regular large-M case. Six large-M shapes reused the same cached object
and HSACO; timings were equivalent. Legacy inherited `bq=4` and prototype
scope text in component JSON are not the actual comparison: explicit
`control_bq=16`, `candidate_bq=16`, `integrated_entry=true` and compiler records
identify the tested path. Preserve original evidence rather than rewrite it.

CPU validation:25 tests,31 subtests; four trace-checker tests. One unrelated
pytest asyncio_mode warning. Production selector still limits original V4,
native TP8 prefill; runtime-M only selects the group16 kernel. Group4/8 and
default-off calls retain the existing implementation. No decode/DSpark/V4.1
or weight-precision changes.

## Evidence and next action

Directory: `.agents/experiments/dsv4_c16_indexer_runtime_service_20260915/`.
Run `analyze.py` only after complete services; it checks owned stop records,
all-rank compile reuse, source/input identity, capacity, shapes and outputs.
Archive88 files,2362930 bytes; SHA256
`13154365c9af75b109146319801f245f571ca7dd3d6ee9d07a3a31438767321f`.

P0 serving gate passed. At this record's creation launcher defaults remain
group4/runtime-off: the final group16/runtime profile is now being measured
with asynchronous markers before editing the source-hashed launcher defaults.
That separate diagnostic is not yet complete. Continue P1 updated stage
budget, then P2 bounded MHC4->8 screen, then independent long-prefix coverage.
