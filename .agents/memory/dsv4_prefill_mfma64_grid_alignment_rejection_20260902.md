# DSV4 prefill MFMA64 grid-alignment rejection (2026-09-02)

## Question

The raw-layout MFMA64 expert kernels use a grid-stride loop.  With the default
gate grid of 416 blocks and 64 output-row tiles, a CTA does not remain on the
same row tile across successive tasks.  This experiment tested whether grids
aligned to the row-tile count improve temporal weight locality at high prefill
occupancy.

## Method

- physical GPU 4 (`HIP_VISIBLE_DEVICES=4`)
- gfx90a, original FP4 weights, raw (non-preshuffled) layout
- M=16384, top-k=6, E=256, H=4096, I=1024, N=4096
- complete routed path: input quant, MFMA64 gate/up, intermediate quant,
  MFMA64 down, and reduction
- initial sweep: two samples per configuration
- confirmation: A/B/B/A, three rounds and three iterations per timing sample
- `amd-smi process --general --sort-by-pid` reported no running processes
  immediately before each GPU experiment

## Initial sweep

| gate blocks | down blocks | median (us) |
|---:|---:|---:|
| 416 | 312 | 204774 |
| 128 | 256 | 288286 |
| 256 | 256 | 278753 |
| 384 | 256 | 228463 |
| 512 | 256 | 237148 |
| 256 | 512 | 268486 |

## ABBA confirmation

| arm | run | trimmed mean (us) |
|---|---:|---:|
| default 416/312 | A1 | 208148 |
| aligned 384/256 | B1 | 231274 |
| aligned 384/256 | B2 | 230863 |
| default 416/312 | A2 | 210785 |

The aligned candidate is consistently about 10% slower.  The first two ABBA
attempts exited during Python import because they used environments missing
AIter/orjson; they launched no GPU work and are excluded.  The accepted ABBA
used `/home/pc/anaconda3/envs/DS/bin/python`.

## Conclusion

Reject grid alignment as a production optimization.  Any cache benefit from
keeping the task stride aligned to output-row tiles is outweighed by reduced
parallelism or worse task distribution.  Do not continue sweeping block counts.
The next high-occupancy kernel experiment must change work decomposition (for
example an expert-row persistent MFMA64 task), rather than only changing grid
geometry.

