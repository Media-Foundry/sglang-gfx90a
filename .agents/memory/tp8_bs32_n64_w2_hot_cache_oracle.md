# TP8 BS32 learned-layer N64 w2-only hot-cache mixed oracle

Date: 2026-08-27

## Scope and method

- Standalone oracle only; no production selector or serving path was changed.
- Baseline: current A4/R2/W8/B832 packed-FP4 down producer with CTA-local LDS
  LUT, followed by the existing fixed-order FP32-partial to BF16 reduction.
- Candidate: identical sorter metadata, output partial layout and reduction.
  Each expert block uses a compact signed-INT8 pre-expanded w2 tensor when its
  expert is in that layer's train-window top 64; all other blocks use the same
  packed-FP4/LDS-LUT path as the baseline.
- Hot experts came only from raw passes 37--100. Benchmark passes came from the
  held-out raw 101--164 window and were selected for proximity to each layer's
  aggregate held-out A4 scan hit, not for favorable latency.
- The three learned layers cover the low, p50 and high ends of N64 scan hit.
- Timing is seven-round ABBA, 100 iterations per leg, with ten warmups.

## Results

| case | layer/pass | hot A4 blocks | packed baseline | mixed N64 | saved | result |
|---|---:|---:|---:|---:|---:|---|
| low | 35/108 | 55/116 (47.41%) | 137.695 us | 142.098 us | -4.402 us | fail |
| p50 | 12/130 | 56/108 (51.85%) | 134.831 us | 140.579 us | -5.748 us | fail |
| high | 16/120 | 62/107 (57.94%) | 138.701 us | 145.888 us | -7.187 us | fail |

Each case passed 100 randomized correctness replays with both the full
`[32,6,4096]` FP32 partial and final `[32,4096]` BF16 output elementwise exact.
None passes the required `>=10 us/layer` saving gate; all regress.

The compact N64 cache tensor is exactly 64 MiB per learned layer. Forty learned
layers would require 2.5 GiB/GCD, excluding allocator fragmentation. The first
allocation/build observed 105.2 ms including allocator cold start and a 384 MiB
reserved-pool growth; subsequent layer cache builds were 2.47--2.63 ms and
increased live allocation by exactly 64 MiB each.

The monotonic regression as hot-block coverage increases is decisive: w2 INT8
pre-expansion doubles weight bytes relative to packed FP4, while the current LDS
decoder has already made unpack relatively cheap. Saving decode instructions
does not repay the extra HBM traffic. Do not implement the N64 cache in
production and do not spend 2.5 GiB/GCD on it. Further routed-kernel work should
keep weights packed and target consumer fusion/vector-load scheduling instead.

Reproduction:

```bash
amd-smi process --general --sort-by-pid -g 0 1 2 3 4 5 6 7
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/bench_dsv4_gfx90a_hot_cache_oracle.py \
  --recorder /tmp/expert_distribution_recorder_1787803355.1855972.pt \
  --analysis .agents/memory/tp8_bs32_diverse_hot_expert_cache.json \
  --warmup 10 --iterations 100 --rounds 7 --correctness-replays 100 \
  --output .agents/memory/tp8_bs32_n64_w2_hot_cache_oracle.json
```
