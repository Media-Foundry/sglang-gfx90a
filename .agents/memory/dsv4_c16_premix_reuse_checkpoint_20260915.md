# Original V4 TP8 C16: FP32 pre-mix row reuse (2026-09-15)

## Accepted outcome

Completed fresh-service A1-B1-B2-A2, three timed waves per leg:

| Leg | Input tok/s per wave | Median |
|---|---|---:|
| A1 | 5939.752 / 5946.332 / 5939.169 | 5939.752 |
| B1 | 6483.176 / 6475.225 / 6477.475 | 6477.475 |
| B2 | 6483.884 / 6476.914 / 6473.467 | 6476.914 |
| A2 | 5939.566 / 5935.061 / 5929.666 | 5935.061 |

Mean of leg medians: **5937.4063 -> 6477.1941 input tok/s (+9.0913%)**.
Mean of leg median-request TTFT: **13.88174 -> 12.74986 s (-8.1537%)**.
Whole-wave median durations: A1/A2 22.06641/22.08385 s versus B1/B2
20.23458/20.23634 s. These are distinct from median-request TTFT.

All four timed legs have exactly 12 logged forwards of four requests with
rounded M32768 (four forwards per wave). Source hashes match across services;
all eight B ranks hit the new path, neither control does. The actual 1M pool
and zero prefix hits pass the analyzer. This improves the already accepted
post-fused4 baseline, not an unoptimized control.

After acceptance, the new flag defaults to1 only in the launcher's combined
TP8 multi-request and prefill-throughput profile (TP8/EP1/no-A2A). Explicit0
is preserved. The ABBA used explicit0/1 before that default-line change;
there is no separate claim of a post-default new-process performance run.
Arithmetic is unchanged from the measured manifest; the kernel file's trailing
blank line was removed after measurement, with no code changes.

## Arithmetic and scope

Four token rows share loads of the FP32 HC projection weight, retaining a
separate accumulator and RMS reduction for each row. Each output column still
uses the literal `[1,1024]` K tile, sixteen K chunks, FP32 multiply/accumulate,
and one wave64. No output-column fusion, FP16 weights, full FP32 activation
expansion, additional workspace, or checkpoint changes are introduced.

The new `gfx90a_mhc_premix_reuse.py` wrapper handles ragged groups by masking
rows 1..3. The existing K2048 override deliberately retains the original
implementation. Unsupported layout/dtype/device also falls back.

`SGLANG_DSV4_PREFILL_MIX_REUSE4` has an independent eager-runner ContextVar.
The conservative scope requires original V4/43 layers/H4096, TP8/attention TP8,
EP1/CP1/PP1, native non-speculative extend, M8192..65536, gfx90a, no rewritten
or TBO batch, and no graph capture. It excludes native decode, speculative
verification/draft, TP4, and V4.1. Exception paths reset both MHC contexts.
The post-reuse decorator also now accepts the runner's `forward_batch=` keyword.

## Integrated component evidence

Physical GCD4 only, before any service ABBA. The initial/timing fixture uses
19 real sampled layer0 residual rows repeated/scaled to the test shape and
the captured FP32 HC Fn. This is not a fresh full-model M32768 capture.
Mutation tests randomize both activation and FP32 Fn copies; original model
files remain untouched. RMS epsilon cycles through 1e-6, 1e-5, and 1e-8.

| M | Control ms | Integrated reuse4 ms |
|---:|---:|---:|
| 128 | 0.082432 | 0.068624 |
| 8192 | 3.761053 | 2.209096 |
| 32767 | 14.974213 | 8.847952 |
| 32768 | 14.892566 | 8.838624 |
| 65535 | 29.777968 | 17.474049 |
| 65536 | 29.517282 | 17.434197 |

All 600 integrated mutations pass bitwise equality and finiteness. Control and
candidate are exact under row permutation at all six shapes. M128 additionally
passes 1000 HIP graph replays; the test explicitly forces scope, while ordinary
runtime dispatch excludes that small M and graph capture. This is not a decode
performance claim. Compiler metadata reports 67 registers through M32768 and
68 at M65535/65536, no spills, and no shared/LDS allocation.

Earlier standalone BM2/BM4 screens and 400 strengthened mutations also pass;
the accepted correctness evidence must refer to the integrated runtime test,
not merely its original experimental copy.

## Service protocol

Experiment: `.agents/experiments/dsv4_c16_premix_reuse_20260915/`.
Three fresh owned services A1 -> B1/B2 -> A2, three timed waves per leg and
one excluded warmup per service. Both arms retain post-fused4 and prefill C4
empty-tile skip. Only mix-reuse4 changes (explicit 0/1).

Original V4 at `/home/pc/models/modelscope`, TP8/EP1/no-A2A/native AR,
1,048,576 logical KV tokens, chunk/max-prefill budget32768, C16 real source
code requests with 131069 input IDs and max_new_tokens=1, no prefix hits.
Runtime source hashes and input manifests are compared across all services.
The analyzer requires actual eight-rank selector hits and identical timed
forward shapes, not merely an environment variable setting.

Each service also runs France before large-prefill warmup, then two independent
C16 x128-output-token quality waves with explicit IDs, fresh cache salts,
input echoes, completion counts, and ID/text consistency checks. Whole-model
repeatability is evaluated separately from local kernel exactness.

`status.py` checks PID birth; `sweep.py` refuses implicit restarts or overwrite;
`analyze.py` requires complete arms and owned shutdown. Its quality report
includes both repeat waves and all cross-arm pairings, not just first waves.

## Service quality and remaining drift

All three France probes answer Paris. All 96 explicit input echoes match,
all quality responses contain 128 completion tokens, and decoding completion
IDs equals returned text. The first B quality wave matches A1 **16/16**.
First-wave cross-process comparisons are A1/A2 15/16 and A2/B 15/16.

Repeat waves are A1 **15/16**, B **15/16**, A2 **16/16**, NOT fully repeatable:

- A1 case8 changes after 40 common tokens, IDs666/2019. This is the same
  alternative wording around `get_available_gpu_memory` seen between controls.
- B case4 changes after 33 common tokens, IDs14114/979, from "key issues visible
  in this excerpt" to "key issues", then a differently structured loader
  analysis. This case does NOT have a matching divergence in these controls.
  Local exactness and a drifting control are not proof of its cause; attribution
  remains unresolved. Do not describe all observed drift as control-identical.
- Comparing every pair of quality waves gives 14..16/16 exact; the analyzer
  preserves all pairings and first-divergence token IDs.

All 16 first-wave candidate excerpts plus the differing second-wave case were
read: coherent, source-code-related analysis without obvious repetition
collapse. This is a 128-token excerpt check, not a comprehensive code-correctness
evaluation. The performance checkpoint accepts small semantic wording drift
under the user's stated policy; it does not claim whole-model determinism or
that earlier compressor/projection/CK atomic drift has been fixed.

## Final validation and checkpoint

CPU scope/dispatcher/launcher/marker/indexer tests: **19 passed**, 23 subtests,
one unrelated pytest asyncio_mode configuration warning. Launcher tests cover
both MHC flags independently, profile conjunctions, explicit0/1, TP4, EP2 and
Mori exclusions. `bash -n` and `git diff --check` pass.

Evidence archive: 86 files, 2,360,255 bytes, SHA256
`bae941831f0b092f9da0714190e3336d4d62e60485c25b0fce8eaeb5c1312d06`.
The archive includes source manifests, timed/quality responses, startup and
selector logs, clean owned shutdowns, and standalone/integrated component
JSON/logs. No model tensors are included. All three services stopped with no
remaining owned children; AMD-SMI subsequently reported no processes on any
of the eight GCDs.

Next: the reviewer's prefill empty-tile suggestion has already been implemented
and is enabled in both arms here. Do not repeat it as a new proposal. Local
multi-query C4 indexer KV reuse is still a distinct candidate; use the measured
~2.64 s C4-logits diagnostic budget rather than claiming MoE dominates the
whole wave. A new profile after these MHC wins should precede any revised
whole-service budget. The MHC win does not imply 6477 tok/s is a hardware limit.
