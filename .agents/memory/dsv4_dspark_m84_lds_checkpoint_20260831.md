# DSV4 TP4 DSpark M84/M96 LDS-unpack checkpoint

Date: 2026-08-31

## Scope

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- TP4 / EP1 / no A2A, original DeepSeek-V4-Flash weights
- DSpark compact verification, gamma 5, causal draft attention
- Forced verify budget fraction: 0.30; graph-tier fill enabled
- CUDA graph request tiers: 1 through 32
- Workload: 32 distinct concrete coding prompts, 256 generated tokens
- Correctness: official France first-nine-token oracle after every service

## Live shape distribution

The existing low-overhead DSpark core observer recorded 249 real decode steps.
The common BS32 window contained 75 steps, all with target graph key M84 and
80 locally scheduled verify rows.  Retirement tiers were mainly M24 (53
steps), M12 (32), and M72 (24).  This proved that M84 is the dominant full-batch
target tier even though compact-mode eager/break segments also log transient
exact-M shapes.

The four M84 BF16 projection GEMMs were tuned independently.  In graph replay,
hipBLASLt saved roughly 30 us per layer across N=256/512/1024/2048, but service
throughput was unchanged because those projections overlap the compressor and
attention branches.  They remain rejected as a production table.

## Target-verify critical path

CUDA-event segment timing at BS32/M84 gave approximately:

| segment | median time |
|---|---:|
| target verify | 96.06 ms |
| draft | 16.37 ms |
| complete speculative step | 115.30 ms |

The events expose synchronization and therefore inflate absolute time versus
the no-observer service, but target verify still accounts for about 83% of the
critical path.

A layer-20 `s_memrealtime` marker then localized target verify without Kineto.
The full-BS32 four-rank spans were approximately:

| layer segment | rank-max range |
|---|---:|
| attention-entry MHC/Norm | 90--96 us |
| attention prepare | 268--279 us |
| attention core | 54--95 us |
| output projection/collective | 166--180 us |
| FFN-entry MHC/Norm | 101--104 us |
| complete MoE | 1.42--1.48 ms |
| routed FP4 expert body | 1.27--1.34 ms |
| routed tail all-reduce | 60--79 us |

Thus the routed FP4 stage alone contributed roughly 56 ms over 43 layers and
was the first DSpark kernel bottleneck.  A full Kineto profile was rejected:
ROCm profiler interposition entered a cross-rank signal wait and the exact
profile service was terminated without retaining results.

## Target-only occupancy recorder fix

`per_pass` expert recording originally intercepted the DSpark draft model's
block-shaped top-k tensor during draft graph capture.  The recorder expects a
target `[tokens, topk]` tensor and failed in `scatter_add_` with a dimension
mismatch.  `DeepseekV4ForCausalLMDSpark.forward` now follows the other MTP and
next-n models and wraps all draft stages in
`expert_distribution_recorder.disable_this_region()`.  Target routes remain
recorded; proposal routes cannot corrupt target occupancy statistics.

Tests:

```text
test_dspark_expert_recorder.py + test_dspark_draft_cpu_lens.py: 2 passed
test_environ.py: 13 passed, 2 subtests passed
```

The fixed service completed target and draft graph capture with recorder mode
`per_pass` and dumped 247 records per rank.

## Real M84 route

Forty full-M84 target passes were analyzed.  Each layer had 504 routed
assignments (84 rows x top-6).

| layer class | active experts | A4 scans | A4 padding | max occupancy |
|---|---:|---:|---:|---:|
| hash layers 0--2, median | 200 | 226 | 400 | 10 |
| learned layers, median | 154 | 202 | 304 | 23 |
| layer 20 representative | 152 | 200 | 296 | 22 |

The route is close enough to the accepted M64 decomposition that the existing
LDS E2M1 lookup is applicable.  M84 had fallen off it only because the selector
hard-coded `num_prefill_tokens <= 64`.

## Isolated M84 oracle

Single variable: A4/R2/G2080/D832, original packed FP4 layout and fixed-order
FP32 reduction; compare direct nibble decode with the existing LDS E2M1 LUT.
Seven samples per arm on physical GCD 4 were elementwise exact.

| stage | no LDS | LDS LUT | change |
|---|---:|---:|---:|
| gate/up | 799.818 us | 529.660 us | -33.8% |
| activation quant | 45.270 us | 40.427 us | -10.7% |
| down | 461.910 us | 380.128 us | -17.7% |
| reduction | 7.616 us | 7.637 us | neutral |
| complete routed | 1284.881 us | 933.031 us | **-27.4%** |

## Service ABBA

Arm A keeps the old 64-row ceiling.  Arm B raises only the ceiling to 96, so
M66/72/78/84/90/96 reuse the same exact LDS kernel family.  Every service ran
two 256-token rounds; round 1 supplies scheduler/step counters.

| order | ceiling | resident tok/s | scheduler tok/s | host step | accept | France |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 64 | 389.855 | 404.051 | 84.824 ms | 1.920 | exact |
| B1 | 96 | 477.729 | 431.103 | 76.203 ms | 1.895 | exact |
| B2 | 96 | 422.885 | 415.677 | 78.992 ms | 1.885 | exact |
| A2 | 64 | 407.088 | 400.382 | 82.693 ms | 1.925 | exact |

ABBA centers:

```text
resident:  398.471 -> 450.307 tok/s  (+13.01%)
scheduler: 402.216 -> 423.390 tok/s  (+5.26%)
host step:  83.759 ->  77.598 ms     (-7.36%)
accept:       1.923 ->   1.890       (-1.68%)
```

The accepted-length movement is small and unfavorable, so it cannot explain
the throughput improvement.  Retain the M96 LDS ceiling in the TP4 BS32
profile; the environment default remains M64 for every other profile.

## Artifacts

```text
/tmp/dsv4_dspark_dynamic_m_state.json
/tmp/dsv4_dspark_segment_profile_state.json
/tmp/dsv4_dspark_target_realtime_marker.log
/tmp/expert_distribution_recorder_1788129217.*_0.pt
/tmp/dsv4_dspark_m84_route_histogram.json
/tmp/dsv4_dspark_m84_lds_oracle.log
/tmp/dsv4_dspark_m84_lds_{A1,B1,B2,A2}.{log,json}
```
