# DSpark M128 two-stage MFMA MHC boundary oracle (rejected)

Date: 2026-09-01

## Candidate

Standalone-only experiment; no production selector imports these files.

1. A two-wave producer tiles 16 tokens and split-K=16, materializes the BF16
   post-combined residual, emits RMS partials, and uses
   `v_mfma_f32_16x16x16f16` for 24 pre-mix dot products.
2. A per-token consumer reduces fixed split slots, performs 20 Sinkhorn
   iterations, weighted residual reduction, and the following RMSNorm.

The old ephemeral real M32 FFN dump had already been removed. The formal run
therefore used deterministic bounded M128 tensors. The large `fn` matrix was
created in FP16 first and promoted to FP32 for the baseline, so both arms saw
the same representable weight values.

## Validation

- Physical GPU 4 only, after `amd-smi process` reported no active process.
- 100 input/residual/post/comb mutations.
- 1000 HIP Graph replays.
- Seven-round symmetric ABBA, 100 replays per timing sample.

The candidate failed numerical equivalence at the first run and throughout
mutation/replay:

| boundary | maximum observed error |
|---|---:|
| residual BF16 | max abs 0.00390625 |
| mixes | max abs 5.49, rel-L2 1.26 |
| post | max abs 1.70, rel-L2 0.52 |
| comb | max abs 0.84, rel-L2 8.64 |
| final layer input | max abs 1.64, rel-L2 0.123 |

The large mix error shows the proposed MFMA lane-to-M/N mapping and/or split
association does not reproduce the reference, rather than exposing a small
FP16-weight rounding difference.

## Performance

| arm | trimmed mean |
|---|---:|
| current Triton decomposition | 114.193 us |
| two-stage candidate | 142.050 us |
| producer alone | 132.262 us |
| consumer alone | 16.661 us |

The complete candidate is 27.856 us slower, a 19.61% regression. It misses the
declared 36-us saving gate even before accounting for its incorrect output.

Static gfx90a resources were healthy but not limiting:

```text
producer: 36 VGPR, 30 SGPR, 0 LDS, 0 scratch, occupancy 8 waves/SIMD
consumer: 44 VGPR, 41 SGPR, 232 B LDS, 0 scratch, occupancy 8 waves/SIMD
```

## Decision

Reject and do not connect to the model. The producer is already slower than
the entire baseline, so correcting the MFMA mapping cannot recover the required
speedup without a different work decomposition. The standalone files are kept
only as a reproducible negative oracle.

Artifacts:

```text
/tmp/dsv4_m128_mhc_two_stage.json
/tmp/dsv4_m128_mhc_two_stage.log
/tmp/dsv4_m128_mhc_two_stage_resource.log
```

