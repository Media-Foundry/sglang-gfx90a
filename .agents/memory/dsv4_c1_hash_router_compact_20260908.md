# C1 hash-router selected projection screen — 2026-09-08

Parent50929613fd. Native baseline3325034 unchanged,1M KV pool retained.
AMD-SMI before isolatedGPU4 test: no foreign owners. Live WAVE64_GEMV=1.
Local config: num_hash_layers3, topk6, scoring_func sqrtsoftplus.
HashTopK's nonsoftmax weights normalize selected scores, so unselected logits
are mathematically unnecessary for that consumer. This is not valid for its
softmax mode or arbitrary consumers of StandardTopKOutput.router_logits.

Compare existing full N256 wave64 BF16 GEMV with N8 (six selected plus two
padding rows), both preselected outside graph and actual index_select inside.
No persistent weight cache in production.100 input/weight/ID mutations:
six selected BF16 logits exact in both candidates;1000 graph replays stable.
No full HashTopK normalization, raw tid2eid load, model or E2E validation.

Five symmetric-order cycles,100 replays/sample, trimmed means:
full2567.308885us, ideal preselected8 7.324286us, gather8+GEMV11.705336us.
Even free gather does not improve this existing kernel geometry; actual gather
adds a launch and copying. Reducing weight bytes alone is not a latency gain.
No production integration. This does not disprove a dedicated indexed/fused
kernel, but three hash layers limit savings and no measured gain justifies it.

Script scripts/rocm/bench_dsv4_c1_hash_router_compact.py; raw samples adjacent.
