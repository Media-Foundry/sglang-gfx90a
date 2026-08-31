# DSV4 DSpark M128 decode-geometry checkpoint (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights, TP4/EP1/no-A2A.
- Physical GCDs `HIP_VISIBLE_DEVICES=4,5,6,7`; standalone oracle on GCD 4.
- DSpark gamma three: resident BS32 verifies exactly M128 target rows.
- 32 distinct coding prompts from
  `.agents/memory/dsv4_tp4_code_32_input_ids.json`.
- Four independent services in A/B/B/A order, two 128-token rounds each.

## Root cause

The accepted TP4 throughput profile capped the grouped-FP4 LDS E2M1 lookup at
M96.  M128 therefore first fell back to direct nibble decode.  Raising the LDS
ceiling from 96 to 128 improved the preliminary gamma-three service:

```text
scheduler: 388.8 -> 441.8 tok/s
host step: 128.93 -> 113.75 ms
resident:  460.9 -> 527.2 tok/s
```

M128 then still selected the generic prefill grid, G416 for gate/up and D312
for down, while the successful M64--M96 decode family used G2080/D832.  This
was a launch-geometry choice only; packed weights, activation quantization,
sort order, fixed-slot FP32 partials and reduction order remained unchanged.

## Real M128 oracle

A target-only `per_pass` recorder captured 32 full M128 passes from real
heterogeneous requests.  The layer-20 route used by the oracle had:

```text
active experts: 125
A4 scans:       250
max occupancy:  117
```

On physical GCD 4, seven-round graph ABBA measured the complete routed stage:

| geometry | latency |
|---|---:|
| prefill G416/D312 | 1361.149 us |
| decode G2080/D832 | 1268.095 us |

The decode geometry saved 93.054 us/layer (7.34%).  It passed 100 randomized
activation/router-weight mutations bitwise for intermediate BF16, group-32
INT8 activation/scales, FP32 partials and final BF16 output.  It also passed
1000 HIP Graph replays with bitwise-stable output.

## Independent-service ABBA

A used M128 LDS with G416/D312; B changed only to G2080/D832. Combined medians:

| metric | A | B | change |
|---|---:|---:|---:|
| scheduler decode | 444.489 tok/s | 466.361 tok/s | +4.92% |
| host speculative step | 112.355 ms | 106.603 ms | -5.12% |
| common-resident | 529.501 tok/s | 549.631 tok/s | +3.80% |
| aggregate HTTP | 448.248 tok/s | 460.577 tok/s | +2.75% |
| mean accepted length | 2.03205 | 2.02109 | -0.54% |

The acceptance movement is small and unfavorable, so it cannot explain the
speedup.  All four services passed the exact France first-nine-token and
semantic Paris oracle.  All 256 coding requests completed 128 tokens with
`finish=length`.

## Decision

Raise the TP4 BS32 profile's LDS ceiling to 128 and enable the strict
`SGLANG_DSV4_GFX90A_M128_DECODE_GEOMETRY` selector there.  Keep the environment
default false and require exactly M128/H4096/top-6 with the original TP4 weight
shapes.  Gamma one remains the service default because optimized gamma three
still has much lower end-to-end throughput; this checkpoint is a reusable
deeper-verification kernel rather than a global gamma switch.

Artifacts:

```text
/tmp/dsv4_gamma3_recorder2/expert_distribution_recorder_*.pt
/tmp/dsv4_gamma3_lds128.json
/tmp/dsv4_m128geom_{b1,b2,a2}.json
/tmp/dsv4_m128geom_{b1,b2,a2}_france.json
scripts/rocm/bench_dsv4_dspark_m128_geometry.py
```
