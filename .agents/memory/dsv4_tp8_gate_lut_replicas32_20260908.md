# Gate LDS lane-replicated LUT32: exact but slower

Baseline59d6aa4a7d, restored uninstrumented PID3277900. FranceC32 restored
32/32, original1M KV pool retained. AMD-SMI confirmed only baseline owners
before the GPU4 standalone test; probe exited normally.

Preserve current A4/R2/W8/G832, gate prefetch, accumulator ownership, SDOT
and fixed reduction. Replicate each256-entry LDS codebook entry32 times and
select replica lane%32, attempting to avoid data-dependent LDS bank conflicts.
Per-CTA LUT rises1KiB→32KiB; this is LDS, not a persistent VRAM weight cache.
Only the standalone module defines SGLANG_FP4_GATE_LUT_REPLICAS_ORACLE.

An exhaustive16-bit address/reconstruction CPU check for8/16/32 replicas
passes, using representative replica lanes and a synthetic byte-pair table.
GPU32-replica full numerical tests pass100 mutations in each of three
synthetic routing distributions: BF16 intermediate/final and FP32 partial
exact, max_abs0, graph replay stable. Not a real routed-input fixture.

Full-chain trimmed microseconds, five ABBA cycles:

|Active/scans|Baseline|LUT32|
|---|---:|---:|
|133/133|275.1275|286.2276|
|104/105|241.6469|255.6458|
|32/61|182.9450|193.9590|

About4–6% latency regression. No E2E integration. This does not prove bank
conflicts were the bottleneck, nor attribute the loss uniquely to occupancy
versus initialization; no counters collected. Eight/sixteen replicas have
only address checks, not performance validation. Keep standard1KiB LUT.

Reproduction: DS interpreter, repository PYTHONPATH, HIP_VISIBLE_DEVICES=4,
`scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --lut-replicas 32`.
