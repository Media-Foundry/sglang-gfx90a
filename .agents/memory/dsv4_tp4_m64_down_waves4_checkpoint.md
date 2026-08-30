# TP4 M64 W4 logical-scale down checkpoint

Date: 2026-08-30

## Component oracle

The real TP4 M64 pass20/layer34 route was tested with original packed FP4
weights, A4/R2, D832, LDS decoding, logical W2 scales and identical fixed-slot
reduction. Gate remains W8; only the down kernel changes W8 to W4.

```text
W8 down: 296.04 us
W4 down: 283.76 us
down improvement: 4.15% / 12.28 us
```

Using W4 for both gate and down was not selected: W4 gate was about 2% slower.

## Service ABBA

Profile: TP4/EP1/no-A2A, original weights, native AR, graph tiers 1 and 64,
64 real heterogeneous requests, 128 generated tokens.

```text
A1 W8:  992.83 resident tok/s
B  W4: 1000.39 resident tok/s
A2 W8:  992.01 resident tok/s
B versus A midpoint: +0.80%
```

Six of eight equal resident windows in B sustained 1016.15 tok/s. Two batch
sampling/scheduler seams were about 953 tok/s. Correctness passed:

- 64/64 next-token IDs exact;
- 64/64 token logprob rows exact;
- 64/64 complete top-5 rows exact;
- France sentinel passed;
- all requests completed at `finish=length`.

Production enables W4 only under the strict existing M64 logical-W2
row-prefetch shape. M32 and every generic/fallback tier retain W8.

Artifacts:

- `/tmp/dsv4_m64_w4_dpp.log`
- `/tmp/dsv4_m64_w8_dpp.log`
- `/tmp/dsv4_tp4_bs64_down_w4_teacher.json`
- `/tmp/dsv4_tp4_bs64_down_w4_b.json`
- `/tmp/dsv4_tp4_bs64_down_w4_a2.json`

