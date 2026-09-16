# Isolated query/K grid-order and K32 screen

No production selector is changed here. kernel.py uses the existing query16
score math, changing only CTA coordinates. test_mapping.py verifies ownership
and AST math identity. screen.py checks group order1/4/8/16/32 on synthetic
packed owner inputs against current runtime-M K16 logits and deterministicTopK.

tile_screen.py additionally tests K32/K64 with fixed16x16 MFMA. K32 passed the
tested score/ID contracts and improved large-shape logits1.64..1.76x; K64 failed
score equality and one membership check, so was not timed/promoted. stress.py
adds large dynamic range, exact ties, mixed prefix, and non-page-aligned widths.
All eight K32 stress cases exact, but very small width513 is slower. Read each
candidate status; screen status complete does not mean every geometry passed.

All results are singleGCD synthetic component graphABBA, not real-model service
throughput. Production remains K16. Actual tested precision/layout is BF16dot,
non-FNUZFP8, preshuffle0. No claim for untested layouts. Runtime-M preserved;
HIP_VISIBLE_DEVICES5 resolves PCI0000:b3:00.0. Model checkpoint unchanged.

See .agents/memory/dsv4_indexer_k32_20260916.md for scope, rejection and next gates.
