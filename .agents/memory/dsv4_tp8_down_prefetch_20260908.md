# TP8 subgroup8 down row-prefetch oracle

Base d42ed1ae10. No production selector changes. Existing TP4 down prefetch
has an explicit subgroup16 static_assert; K256 needs subgroup8. Copied into
independent gfx90a_fp4_tp8_down_prefetch_oracle.cuh and changed only that guard
and its comment, not the original TP4 header. Reduction bounds, assignment
mapping and scale addressing already derive from K/32. No logical-scale cache,
weight repack, or new persistent allocation. Tested existing shuffled scales.

GPU4 only, amd-smi no external PIDs; retained service2821603 idle. Synthetic
transient gate/down weights ~384MiB, freed on exit. No HTTP benchmark overlap.
Standalone script --down-prefetch holds accepted gate-prefetch fixed on both
arms and changes only down. Post-sort chain includes gate/intermediate quant/
down/fixed-slot reduction; excludes initial quant/sorter. Five ABBA cycles,
100replays per sample,20warmups. Original TP8 A4/R2/W8/G832/D832 geometry.

Second run explicitly checks FP32 partials (first run only checked final BF16):

|Active experts / scans|A chain us|B chain us|Saved us|
|---|---:|---:|---:|
|133/133|274.9263|265.7904|9.1359|
|104/105|241.2568|235.1578|6.0990|
|32/61|182.7717|179.7531|3.0186|

All three distributions: gate BF16, down FP32 partial and final BF16 exact
100/100 under mutation; max final error0; candidate final stable across ten
replays per mutation (1000total). No full-model semantic claim from synthetic
weights. Raw samples and witnesses in adjacent JSON. First independent timing
run also saved~3.2–8.9us. Not an E2E result; do not add component gains to rates.

Next: default-off TP8 native M32 down selector, retaining the original scale
layout and fixed reduction; service ABBA with gate-prefetch ON in both arms.
Production remains unchanged until scope tests and real C32 correctness pass.
