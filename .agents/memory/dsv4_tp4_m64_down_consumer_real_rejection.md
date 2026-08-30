# TP4 M64 real down-consumer rejection

Date: 2026-08-30

## Why the earlier service result was a false A/B

The production selector for both the M32 and M64 down-consumer experiment
required:

```python
quant_info.w2_weight.shape == (256, 4096, 128)
```

That is the packed K256 W2 shape.  The TP4 service uses K512 and its actual
packed W2 shape is:

```text
(256, 4096, 256)
```

Consequently `SGLANG_DSV4_GFX90A_M64_DOWN_CONSUMER=1` could not select the
consumer on TP4.  The old service numbers compared the same fallback path on
both sides and must not be cited as evidence that the fused consumer was
neutral.  The old standalone consumer also fixed W8 and used the shuffled
scale lookup, so it did not reproduce the accepted TP4 M64 logical-scale W4
path either.

## True TP4 baseline and candidate

The replacement standalone oracle used the real M64 route from recorder
pass 20, layer 34:

```text
tokens/top-k:       M64 / T6
active experts:     146
A4 expert blocks:   174
packed W2:          [256, 4096, 256] uint8 (K512)
logical W2 scales:  [256, 4096, 16] uint8
```

Baseline A exactly reproduced the accepted chain:

```text
BF16 intermediate
-> gfx90a_int8_group32_quant
-> logical-scale W2 row-prefetch A4/R2/W4/D832/LDS-unpack
-> FP32 [M,T,N] partial
-> unchanged fixed-order reduction
```

Candidate B kept the same packed original FP4 weights, logical E8M0 scale
indexing, SDOT arithmetic, FP32 partial and fixed reduction.  Each CTA
quantized its expert A4 block from BF16 into LDS and immediately consumed the
K512 activation.  The sweep covered W4/W8 and 4/8/12/16 CTA shards per expert
block.

## Correctness

All eight geometries passed:

```text
100 mutations of BF16 intermediate, packed W2 bytes, logical scales and
router weights: partial bitwise exact, reduced BF16 output bitwise exact

1000 HIP Graph replays: bitwise stable
```

There was no numerical first divergence.

## Seven-round ABBA result

Times include quantization, down partial production and the unchanged fixed
reduction.

| waves | CTA/block | A median | B median | B vs A | trimmed delta |
|---:|---:|---:|---:|---:|---:|
| 4 | 4  | 278.283 us | 342.378 us | +23.03% | +23.03% |
| 4 | 8  | 278.531 us | 317.348 us | +13.94% | +13.99% |
| 4 | 12 | 278.477 us | 303.865 us | +9.12%  | +9.09%  |
| 4 | 16 | 278.523 us | 298.028 us | +7.00%  | +6.97%  |
| 8 | 4  | 278.446 us | 323.598 us | +16.22% | +16.25% |
| 8 | 8  | 278.393 us | 300.840 us | +8.06%  | +7.99%  |
| 8 | 12 | 278.517 us | 296.844 us | +6.58%  | +6.56%  |
| 8 | 16 | 278.544 us | 294.694 us | +5.80%  | +5.81%  |

The best candidate was still 5.8% slower than the true accepted baseline,
far from the required at-least-10% complete-stage improvement.

All HSACO variants stayed within the resource gate:

```text
VGPR:   49--50
spill:  0
scratch: 0
LDS:    3376 bytes
```

## Root cause and disposition

The baseline quantizes every `[token, top-k, K512]` assignment once.  The
candidate shards N4096 rows across 4--16 CTAs for every A4 expert block, but
each shard CTA needs its own LDS and therefore repeats the same four K512
quantizations.  Removing the global quant tensor and one launch does not pay
for 4--16 copies of quantization plus the much larger grid.  More CTA shards
hide the down work better, which explains the monotonic improvement, but even
CTA16 cannot recover the duplicated producer cost.

Reject this CTA-local consumer architecture.  A future fused design needs a
single quant producer per A4 block with communication to row consumers, or a
work decomposition in which one CTA owns both the quantized A4 activation and
enough output rows to amortize it.  Do not re-enable the current service flag
for TP4 merely by relaxing the shape guard.

Raw run log during the experiment:
`/tmp/tp4_m64_down_consumer_real_oracle_full.log`.
