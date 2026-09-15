# Native H8 prefill: geometry/lifetime variants rejected; real pair count small

Production stays7953.956103 input tok/s, original V4/TP8/AR/1M KV. No model,
launcher or production attention kernel changes. PhysicalGCD4 only after services
stopped. This turn's progress is measured negative results and a real-data bound.

Current two-bank Triton H16/K16/one-wave kernel was challenged on synthetic
M8192/H8/D512 C128 and varied-C4 causal8K fixtures:

- LogicalH8/one-wave: C4 5.935→96.341ms, heavy compiler-reported spill count1081;
  output not bit-exact (max_abs0.001953125). C128 also~16x slower.
- H16/two-wave: C45.934→9.969ms, C1281.985→3.005ms, five mutations bit-exact.
- ReloadQ inside each KV tile, volatile load to prevent hoisting:
  C45.927→11.266ms, C1281.983→3.512ms, five mutations bit-exact.
  Registers354→416 and LDS16→32KiB, not the intended lifetime/resource reduction.

Each component measured three ABBA cycles with five calls/event. All rejected;
do not expand to M32768/service or repeat these as unexplored settings. No speed
improvement from these tests. Compiler register counts are not occupancy counters.

Before writing another Q2xH8 kernel, replayed real layer20 indexer selection from
SHA-verified M32767 first-four-request fixture of the C16 code wave. Full logical
IDs preserved; three replays exact. Comparing corresponding K16 tiles without
reordering selection gives52855 compatible prefix tiles,1144 inferred compatible
SWA extend tiles, out of1181528 original query-tile executions: **4.5703%**
ideal removed-work bound. In saturated512-key rows only10775 of786592 prefix
tiles are shareable; most sharing is early trivial-query work.

This is only one real C4 layer/forward, not a whole-model latency bound. It assumes
free compatibility checks, no occupancy cost, and known zero-prefix flat SWA
window semantics. No fresh main-attention Q/K capture or pair kernel was run.
It does not disprove union schedules with changed softmax grouping or sharing
patterns in other layers, but does not justify immediate general-pair development.

Artifacts `.agents/experiments/dsv4_c16_attention_shape_20260915/`: screen/reload
scripts, candidate Q-reload kernel, output JSONs, overlap.py, full logical IDs in
evidence archive, README and archive manifest. The source query/KV fixture remains
local at the previously recorded capture-v2 path, SHA recorded in overlap.json.

Next plausible bounded opportunity: accepted query-owner still computes every
rank's full query producer before splitting logits/Top-K. Current profile has
0.627s/wave query production. Need a real full-M versus owned-row producer oracle,
including changed-GEMM-shape numerical checks, before integration. Compressor/KV
must remain complete; no implementation or speed claim for that direction yet.
