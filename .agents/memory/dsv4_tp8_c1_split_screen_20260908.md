# Fused C1 KV split screen — 2026-09-08

Base b5f127e125. Isolated GPU4 after AMD-SMI audit found no foreign GPU PID.
Production PID3277900, TP8 native EP1 and 1048576-token KV pool unchanged.
No production selector or default edited.

Shape T1/H8/D512 BF16 unified KV, fused inverse RoPE, blockH16/blockK16,
four waves/stages2. Compare production heuristic64 with32/16 splits.
Five symmetric-order rounds (64,32,16,16,32,64),100 graph replays/sample,
trim minimum/maximum. Synthetic uniform timing contexts, not service E2E.

| KV items | split64 us | split32 us | split16 us |
|---|---:|---:|---:|
|128|17.172|17.229|16.582|
|256|17.164|17.294|16.877|
|640|17.519|22.522|28.491|

Each context additionally changes Q/KV/sink/indices/positions and valid length
100 times, including ten empty cases. All outputs finite.1000 final-state
graph replays stable per profile. Split changes alter floating-point reduction:
exact counts for32/16 are100/92 at128,94/92 at256,74/42 at640.
Largest absolute difference0.00390625, maximum relativeL2 about0.00169.
No service logits or generated-token parity claim.

Retain64: only sub-microsecond short-row savings, severe longer-row regression,
and changed reduction. Graph-time split choice cannot safely use padded index
capacity as actual sequence length. Do not enable a global16/32 selector.
Dynamic length-dependent strategies would be a different implementation and
need explicit numerical and workload coverage, not justified by this screen.

Script scripts/rocm/bench_dsv4_tp8_c1_kv_splits.py; adjacent JSON keeps raw samples.
