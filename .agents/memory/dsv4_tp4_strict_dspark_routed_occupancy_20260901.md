# Strict TP4 DSpark routed occupancy and stage budget (2026-09-01)

## Capture

- DeepSeek-V4-Flash original weights, physical GCDs 4--7.
- TP4 / EP1 / no A2A, DSpark gamma 3, compact verify tiers 64/96/128.
- Forced verify-budget fraction 0.6.
- Both M64 and M128 anchor-only switches disabled: every real verify row ran
  the complete routed expert path.
- 32 distinct code/chat requests, 128 generated tokens, per-token expert
  recorder enabled only for this diagnostic run.
- France first-nine exact; all 32 requests produced 128 tokens and
  `finish_reason=length`.

Raw recorder files are under `/tmp/dsv4_strict_occ/`; the derived summary is
`/tmp/dsv4_strict_occ_summary.json`.

## Physical tier frequency

The target verifier did not spend most of its time at M128:

| physical recorded rows | forward records |
| ---: | ---: |
| 64 | 60 |
| 96 | 85 |
| 128 | 1 |

There were additional batch-drain records below M64. A strict kernel effort
must therefore prioritize M64 and M96; an M128-only kernel cannot materially
move this workload.

## Learned-router occupancy

Values below aggregate layers 3--42 over the recorded target-verify passes.
`repeat chunks` counts every A4 chunk after the first chunk of the same expert.

| tier | active experts mean | A4 scans mean | repeat chunks mean | repeat/scans | useful assignments/A4 capacity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| M64 | 54.55 | 123.39 | 68.85 | **55.79%** | 77.80% |
| M96 | 89.40 | 122.41 | 33.00 | **26.96%** | 65.05% |
| M128 | 133.68 | 255.40 | 121.73 | **47.66%** | 75.18% |

The M96 means include graph-tier rows that are padding/sentinels; full-M96
records have 576 valid assignments. A representative full-M96 learned route
(record 12, layer 34) has 145 active experts, 212 A4 scans and max occupancy
36. A representative M64 route (record 130, layer 6) has 46 active experts,
119 A4 scans and max occupancy 34.

Hash-router layers 0--2 are more dispersed at M96/M128 and must be reported
separately by any candidate rather than hidden inside a learned-layer mean.

## Current full-stage graph cost

Using the production A4/R2/LDS-unpack kernels, production block counts and the
representative recorded distributions, standalone graph ABBA measured:

| tier | gate | quant | down | reduce | complete routed stage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| M64 | 357.5 us | 42.1 us | 284.6 us | 5.5 us | **658 us** |
| M96 | 577.0 us | 42.2 us | 457.0 us | 9.9 us | **1060 us** |

The older A1/A2/A4 multi-launch bucket implementation was again slower on both
shapes; this is only a baseline measurement, not a proposal to revive it.

## Decision gate for same-expert wave pods

The proposed CTA wave-pod layout keeps each wave's existing A4 arithmetic and
fixed output slot, but places equal-expert chunks at the same output-row tile
in sibling waves so cache hierarchy can reuse packed weight rows. It may not
use LDS exchange, CTA barriers, host occupancy synchronization, or change the
reduction order.

Continue only if:

- learned-route geomean improves at least 22%;
- rank-max/full-stage saving reaches about 255--270 us per layer;
- hash layers do not regress more than 5%;
- 100 metadata/input/router-weight mutations and 1000 graph replays remain
  bitwise stable;
- profiler weight-read requests fall at least 15% on covered chunks.

M96's 27% repeat coverage makes the gate tight: a candidate with appreciable
scheduling overhead cannot win. M64 has much more reuse headroom and is the
first shape to validate.
