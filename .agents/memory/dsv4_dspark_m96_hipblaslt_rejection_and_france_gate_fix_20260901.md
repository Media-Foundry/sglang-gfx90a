# DSpark M96 hipBLASLt rejection and France gate fix (2026-09-01)

## Scope

- Model: original `/home/pc/models/modelscope` DeepSeek-V4-Flash checkpoint.
- Hardware: physical GCDs 4,5,6,7; TP4/EP1/no-A2A.
- DSpark: gamma 3, resident BS32 target M128 and draft M96.
- Workload: 32 heterogeneous token-ID prompts selected once with seed
  `20260901`, then materialized at `/tmp/dsv4_2k_seed20260901.json`.
- Canonical selected-workload digest reported by the harness:
  `6699bf7e5153eaf6625f72954d8cad92064c944067874a6297bb780cf2958f5c`.

## Offline tuning

The draft graph emitted untuned BF16 M96/K4096 projections. AIter's
hipBLASLt tuner selected the following 104-CU gfx90a solutions:

| N | solution | tuner us | graph default us | graph candidate us |
|---:|---:|---:|---:|---:|
| 256 | 5037 | 14.9826 | 21.4945 | 13.4894 |
| 512 | 4446 | 20.6382 | 30.1363 | 19.1299 |
| 1024 | 5060 | 31.2290 | 37.0011 | 30.6646 |
| 2048 | 4854 | 41.3894 | 43.0800 | 41.2137 |

All four candidates passed the tuner's error threshold. Across 100 random
mutations they were not bitwise equal to the current rocBLAS/torch path; max
relative L2 was `6.97e-5` to `1.25e-4`. The test branch therefore used a
DSpark-module ownership marker plus exact gfx90a/BF16/M96/K4096/N guards. It
cannot be reached by target verification or native AR. A per-N mask supports
isolation: 1=N256, 2=N512, 4=N1024, 8=N2048.

## Service result: rejected

The combined high-value mask 3 (N256+N512) retained BS1 France exact 5/5,
but lost end-to-end throughput despite the standalone kernel wins:

| arm | resident BS32 tok/s, five 32x1024 rounds | trimmed center | mean acceptance |
|---|---|---:|---:|
| control | 1530.35, 1495.78, 1510.71, 1538.03, 1508.42 | 1516.49 | 3.5530 |
| mask 3 | 1513.71, 1464.84, 1450.17, 1477.41, 1443.58 | 1464.14 | 3.5416 |

The candidate regressed the trimmed center by 3.45%. The most plausible
explanation is graph-level resource/scheduling interaction: the isolated
projection savings total only tens of microseconds, while a different
hipBLASLt kernel changes occupancy and overlap with the surrounding draft
attention graph. Keep the diagnostic branch default-off; do not promote any
of these solutions into the TP4 BS32 profile.

Artifacts:

- `/tmp/dsv4_dspark_m96_bf16_tuned.csv`
- `/tmp/dsv4_m96hipb_control_A1_1024_r5_allow.json`
- `/tmp/dsv4_m96hipb_mask3_B1_1024_r5_allow.json`
- `/tmp/dsv4_m96hipb_control_A1_france_bs1_r5_fixed.json`
- `/tmp/dsv4_m96hipb_mask3_B1_france_bs1_r5.json`

## France semantic gate bug

The harness previously defined semantic correctness only as the common prefix
plus token 11111 (`" Paris"`) in the first 16 completion tokens. The historical
exact response uses a different tokenization:

```text
[671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]
-> "The capital of France is **Paris**.<eos>"
```

Consequently `france_first9_exact=True` could incorrectly coexist with
`france_semantic_paris=False`, causing valid control runs to abort. The fixed
gate accepts either the historical exact sequence or the alternate prefix +
token-11111 form. Direct unit assertions cover exact, alternate, and negative
examples. With the fix, both control and mask-3 services passed BS1 exact 5/5.

BS32 still showed genuine greedy trajectory variation (control semantic 3/5,
mask-3 semantic 4/5). Therefore concurrent performance A/B must report the
semantic rate but should also require a separate BS1 exact gate; do not discard
all timing data after the first concurrent semantic divergence.
