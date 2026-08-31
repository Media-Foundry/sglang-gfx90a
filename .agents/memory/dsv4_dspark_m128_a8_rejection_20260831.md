# DSV4 DSpark M128 A8 grouped-FP4 rejection (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights and TP4 expert shapes.
- DSpark gamma-three M128 target-verification route.
- Real 32-request heterogeneous target recorder:
  `/tmp/dsv4_gamma3_recorder2/expert_distribution_recorder_1788146391.4818711_0.pt`.
- Layer 20, physical GCD 4, packed FP4 weights, LDS E2M1 decode, group-32
  INT8 activations, fixed-slot FP32 partials and BF16 reduction.

The existing formal M128 geometry oracle gained an optional `--screen-a8`
mode.  It compares production A4/R2/W8/G2080/D832 with A8/R1/W4 candidates;
the two forms retain the same number of gate/up accumulators per wave.

## Route

```text
active experts: 125
maximum occupancy: 117
A4 scans: 250
A8 scans: 167
```

A8 therefore removes 33.2% of the theoretical expert weight scans.

## Correctness

All candidates passed:

- 100 randomized activation/router-weight mutations, bitwise exact at the
  gate intermediate, INT8 value/scale, FP32 partial and final BF16 output;
- 1000 HIP Graph replays with bitwise-stable output.

## Seven-round result

| geometry | complete routed stage | versus A4 |
|---|---:|---:|
| A4/R2/W8 G2080/D832 | 1267.329 us | baseline |
| A8/R1/W4 G1664/D832 | 1290.128 us | +1.80% latency |
| A8/R1/W4 G2080/D832 | 1286.843 us | +1.54% latency |
| A8/R1/W4 G1664/D1248 | 1360.512 us | +7.35% latency |
| A8/R1/W4 G2080/D1248 | 1357.113 us | +7.09% latency |

The existing A4 decode geometry also reproduced its prior win over the
prefill grid: 1358.310 -> 1267.329 us (-6.70% latency).

## Decision

Reject A8 for M128 and do not wire a gamma-three production selector.  Fewer
weight scans do not compensate for A8's longer dependency chain and reduced
latency hiding, even when rows-per-wave is reduced to one.  Together with the
M64 A8 evidence, fixed A8 grouping is closed for the current sdot4 kernel.

Artifact: `/tmp/dsv4_m128_a8_oracle.log`.
