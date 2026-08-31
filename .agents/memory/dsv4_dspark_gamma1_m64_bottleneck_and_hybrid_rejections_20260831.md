# DSV4 TP4 DSpark gamma-one M64 bottleneck and rejected hybrids (2026-08-31)

## Scope and invariant

- Physical devices: `HIP_VISIBLE_DEVICES=4,5,6,7`; single-GPU kernel oracles use physical GPU 4.
- Model: original DeepSeek-V4-Flash safetensors, TP4/EP1/no-A2A.
- Correctness gate: official France first-nine-token oracle plus 32 distinct fixed coding prompts.
- Stable service checkpoint remains gamma one (`60b3832844`): roughly 760--776 resident tok/s and 676--689 aggregate tok/s on 32 diverse 256-token requests.

## Gamma-one critical path

A low-overhead DSpark segment observer collected 69 full BS32 steps. The target verify graph is fixed at M64.

| Segment | Trimmed time | Share of 72.44 ms step |
|---|---:|---:|
| Target verify, M64 | 63.651 ms | 87.9% |
| Draft | 6.332 ms | 8.7% |
| Remaining host/scheduler work | about 2.46 ms | 3.4% |

Layer-20 rank-max markers measured roughly 1.43--1.45 ms/layer:

- routed FP4 body: 0.65--0.70 ms (largest component);
- attention prepare: 0.25--0.26 ms;
- output/projection/collective: 0.16--0.17 ms;
- both MHC sides: about 0.19 ms total;
- routed tail all-reduce: about 0.05 ms.

Artifacts:

- `/tmp/dsv4_dspark_gamma1_segments_server_info.json`
- `/tmp/dsv4_dspark_gamma1_m64_marker.log`

## Real M64 expert occupancy

The recorder captured 196 forwards, including 67 full-M64 target passes. For learned-router layers, medians/means were:

- active experts: about 107.38;
- A4 weight scans: about 150.75;
- hypothetical A8 scans: about 120.33;
- maximum expert occupancy: about 33.23;
- expert counts by run length: 38.95 at one, 23.15 at two, 21.8 at three-to-four, and 23.1 above four;
- about 61.4% of assignments belong to experts whose occupancy exceeds four.

Layer 20 had median active experts 102, A4 scans 147, A8 scans 116, and maximum occupancy 58. Hash-router layers stay much more dispersed; A8 cuts their scan count by only about 6%.

Recorder artifacts:

- `/tmp/expert_distribution_recorder_1788135603.386394_0.pt`
- three peer-rank files with the same prefix.

## Rejected A4/A8 occupancy hybrid

Oracle: `scripts/rocm/bench_dsv4_dspark_gamma1_m64_a4a8_hybrid.py`.

It assigns occupancy <=4 to A4 and occupancy >4 to A8, while sharing quantization and the fixed-order reduction. On real pass 93/layer 20:

- 113 active experts;
- baseline A4 scans 160;
- hybrid scans: low-A4 89 plus high-A8 40 = 129;
- high bucket contained 229 assignments.

Correctness passed 100 input/route mutations exactly and 1000 HIP graph replays bitwise. Performance nevertheless regressed:

- baseline A4: 671.172 us;
- best hybrid (`L1040_624_H2080_1248`): 752.999 us, +12.2% latency;
- other hybrid geometries: 758--785 us.

Conclusion: fewer theoretical weight scans do not compensate for the extra launch/task decomposition and A8 accumulator pressure. Do not add an A4/A8 service selector. The next oracle must change ownership/work decomposition while retaining A4 accumulators.

## Rejected gamma-two compact-budget service

Service configuration:

- `SPECULATIVE_DSPARK_BLOCK_SIZE=2`;
- `SGLANG_RAGGED_VERIFY_MODE=compact`;
- existing TP4 BS32 profile and graph-tier alignment;
- same-service ABBA on forced budget fractions 0.50 / 0.75 / 0.75 / 0.50;
- two rounds of 32 distinct coding requests, 128 generated tokens, per arm.

Every runtime change passed the France exact oracle. All 256 diverse requests completed with `finish=length`.

Sorted four-round results:

| Budget | Aggregate tok/s | Resident tok/s | Scheduler tok/s | Host step | Acceptance length |
|---|---|---|---|---|---|
| 0.50 | 485.6, 493.1, 497.1, 553.3 | 395.5, 411.6, 471.6, 473.9 | 547.7, 574.5, 575.5, 683.0 | 80.17--80.37 ms | 1.516--1.817 |
| 0.75 | 462.9, 484.6, 489.2, 493.3 | 381.5, 402.9, 422.8, 431.5 | 534.8, 559.7, 569.1, 583.9 | 86.38--91.31 ms | 1.615--1.672 |

Fraction 0.50 is less bad than 0.75, but both are substantially slower than gamma one's approximately 72.3 ms host step and 760--776 resident tok/s. Do not replace the gamma-one checkpoint with compact gamma two. The runtime budget cutoff does not reproduce gamma one's cheaper fixed-M64 target path.

Raw outputs:

- `/tmp/gamma2_compact_a1.json`, `/tmp/gamma2_compact_a2.json`
- `/tmp/gamma2_compact_b1.json`, `/tmp/gamma2_compact_b2.json`
- matching `*_france.json` files.

