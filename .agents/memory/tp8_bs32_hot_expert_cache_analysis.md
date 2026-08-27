# TP8 BS32 diverse held-out hot-expert cache analysis

Date: 2026-08-27

## Method

- Input: `/tmp/expert_distribution_recorder_1787803355.1855972.pt`, containing
  168 complete BS32 passes.
- Drop the first 32 complete passes as warmup.
- Use raw passes 37--100 only to choose each layer's hot experts, then evaluate
  on held-out raw passes 101--164.
- Each layer has exactly `64 * 32 * 6 = 12288` held-out assignments.
- Count A4 weight scans as `ceil(occupancy / 4)` per expert and pass. Do not use
  assignment hit rate as a proxy for weight-scan reuse.
- Logical w13/w2 byte hit uses the TP8 H4096/I256 packed-FP4 shapes. It is
  kernel-requested logical traffic, not a hardware-counter HBM measurement.

## Results

| top N | all assignment hit | all A4 scan hit | learned A4 scan hit | hash A4 scan hit | w2 INT8 cache/GCD |
|---:|---:|---:|---:|---:|---:|
| 8 | 19.96% | 11.18% | 11.26% | 10.20% | 0.336 GiB |
| 16 | 31.75% | 19.16% | 19.30% | 17.40% | 0.672 GiB |
| 32 | 46.13% | 31.59% | 31.90% | 27.76% | 1.344 GiB |
| 64 | 63.99% | 51.09% | 51.73% | 43.23% | 2.688 GiB |

At N=32 the assignment hit appears to cross the earlier 46% continuation
threshold, but the relevant A4 scan/logical-weight-byte hit is only 31.59%.
Therefore N<=32 fails the gate. N=64 is the only tested size that passes for
all/learned layers. Hash layers 0--2 still fail at N=64, so a learned-only N=64
prototype would save about 0.188 GiB/GCD relative to caching all 43 layers.

The w2-only estimate assumes a signed-INT8-expanded codebook and reuses existing
scales: `4096 * 256 = 1 MiB` extra per expert-layer. It is an implementation
capacity estimate, not proof of speedup. Any prototype still needs an ABBA
microbenchmark with measured HBM traffic and exact output checks.

## Reproduction

```bash
/home/pc/anaconda3/envs/DS/bin/python \
  .agents/memory/analyze_tp8_bs32_hot_expert_cache.py \
  /tmp/expert_distribution_recorder_1787803355.1855972.pt \
  --output .agents/memory/tp8_bs32_diverse_hot_expert_cache.json
```

The JSON contains the exact selected raw indices, per-layer hot expert IDs,
assignment and A4 scan counts, p50/p95 per-layer statistics, logical byte counts,
and cache-capacity estimates.
