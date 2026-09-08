# Current TP8 M32 wo_b API screen: no gain

Native TP8 decode optimization follow-up, 2026-09-08. No serving code changed.
Service stayed on its baseline configuration with a 1M token pool.

Historical wo_b tuning used N4096/K2048. The current TP8 shape is
M32/N4096/K1024, so those old negative results were not treated as direct proof.
The existing wo_a screen now accepts `--projection wo_b` and tests against
F.linear, the current cached-BF16 M32 fallback. The wave64 GEMV path only
accepts M<=4 and the Triton GEMV only M1.

Physical GPU4; AMD-SMI ownership checked before GPU initialization. Eight real
FP8 checkpoint weights were expanded only for the rank-0 column shard, with
128x128 scales and BF16 rounding. Layer20's shard-only expansion was separately
checked exactly against full expansion followed by slicing on CPU.
Total explicit candidate weight storage is 64 MiB, temporary, not a new model
cache. Input is the first32 rows of a real prefill wo_a dump, not a fresh
diverse-decode fixture. Therefore results are component screening only.

| API | Baseline median us | Candidate median us | Exact mutations |
| --- | ---: | ---: | ---: |
| einsum | 16.77835 | 16.76755 | 100/100 |
| mm | 16.76755 | 16.79275 | 100/100 |
| AIter tgemm | 16.76555 | 16.78194 | 100/100 |

Each comparison used seven multi-weight graph ABBA cycles. All candidates
passed 1000 graph replays bitwise stable. AIter explicitly logged Torch
solution 0 fallback, not a CK kernel. No meaningful advantage; no E2E candidate
introduced. This does not reject every CK or hipBLASLt implementation of this
shape, and no exhaustive solution scan was performed in this experiment.

Artifacts: adjacent JSON retains samples, shape, source and runtime versions.
Command:

```sh
HIP_VISIBLE_DEVICES=4 /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/bench_dsv4_tp8_woa_gemm_screen.py \
  --service-pid 3130584 --projection wo_b \
  --output /tmp/dsv4_tp8_wob_gemm_screen_20260908.json
```

Next decode work should target exposed routed/collective latency or a concrete
new work decomposition, not ordinary output-GEMM API substitution. Active-query
compaction remains a separate prefill opportunity; short C1/C32 decode below
the indexer threshold cannot derive its speedup from that optimization.
