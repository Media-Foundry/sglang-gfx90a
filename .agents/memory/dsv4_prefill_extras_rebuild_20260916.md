# Clean rebuilt prefill extras reproduce 8.67k; real CK writeback counters

This turn made progress: packaged both experiment build contracts independently
of historical AIter Ninja caches, and verified the resulting artifacts in a new
full service ABBA. No global default changed. Original V4, TP8/EP1, native AR,
C16x8K distinct real code prompts, 131069 input tokens/wave, zero prefix hits,
one output token, chunk32768, and 1048576 logical KV tokens throughout.

## Build contract and correctness

`scripts/rocm/build_dsv4_prefill_extras.py` compiles the packaged unique-slot CK
and direct-HIP IPC sources with an explicit ROCm compiler. It never reads old
build.ninja or edits installed AIter/CK. Supported header hashes fail closed;
the unique Set and explicit valid-store guard live in private header overlays.
Transitive non-system dependencies, compiler/Torch/HIP/ABI, exact commands and
module hashes are recorded. Output directories cannot be reused/overwritten.

Local accepted build: experiments/dsv4_prefill_extras_build_20260916/build-v1.
ROCm compiler HIP7.15.26333 / clang23; Torch2.12.0a0+git78258b9, HIP7.14.60850,
CXX11 ABI1. Environment PATH also has a different conda/pip HIP SDK; use /opt/rocm
explicitly. Three CPU build-contract tests pass (pytest reports an unrelated
unknown asyncio_mode config warning).

Rebuilt unique CK: real M8192 and M32767 fixtures, five mutations each, partial
and BF16 output byte-exact to old fixed-slot CK, plus30 poisoned-output replays
per shape. Old fixed-slot, not atomic summation, is the exact reference.
Rebuilt direct-HIP IPC:80 distinct-sink peer mutation checks exact, explicit
peer close successful. Fresh combined service:1312 byte-exact attention
comparisons across41 eligible layers x8ranks x4forwards, France passes.

## Formal old-binary versus rebuilt-binary ABBA

Both arms already have corrected local sinks, H16 and unique-slot CK enabled.
This is rebuild reproducibility, NOT another H8/H16 optimization comparison.

| Leg | Median input tok/s |
| --- | ---: |
| A1 old artifacts |8671.017926|
| B1 rebuilt artifacts |8677.228768|
| B2 rebuilt artifacts |8664.214563|
| A2 old artifacts |8680.545304|

Three waves per leg; fresh processes for A1/B/A2, B1/B2 adjacent in one process.
Old center8675.781615; rebuilt center8670.721665; delta -0.05832%.
Control drift+0.10988%. No discernible service regression or new speedup.
Common source hashes and prompt manifests unchanged throughout.

Every process then runs four C16x128-token quality waves. All16 requests match
across all12 waves and both artifact families:192 responses,24576 output tokens.
Prompt echoes, request IDs, exact lengths and decoded text are checked, not empty
arrays. This repeats the bounded fixed-workload result; arbitrary batching,
prefix lengths, logits and complete semantic correctness remain unproven.

All service processes were stopped by the owned lifecycle after their arm.
Builds and evidence are under experiments/dsv4_prefill_extras_build_20260916.
The experiment still consumes six-slot FP32 partial scratch and512MiB/rank
direct-HIP peer storage;1M pool was verified, not inferred from env settings.

## Hardware counters: isolated historical real M32767 stage2

Artifacts: experiments/dsv4_ck_counters_20260916. Physical GCD6 only, no live
TP8 service. The fixture is historical real original-V4 routing, not a new
corrected-sink capture. Three repetitions in each of two counter passes.
SDK-selected ROCTX regions contain just unique CK stage2 and fixed reducer.
The profiled result is byte-exact to the prior fixed-slot reference.

| Kernel | Profiled ms | HBM fetch (KiB) | HBM write (KiB) |
| --- | ---: | ---: | ---: |
| unique CK stage2 |6.86–6.88|656970.58|3143477.33|
| scalar fixed reducer |3.97|3145633.00|261776.33|

CK resource metadata:112VGPR,64AccVGPR,32KiB LDS,0scratch. Reducer:20VGPR,
4AccVGPR,0LDS,0scratch. CK instruction counts52,494,336 MFMA /185,427,456 VALU;
reducer83,896,832 VALU. These are counts, not utilization percentages.

The large cost is approximately3GiB partial write followed by3GiB reducer read;
CK HBM fetch is only about0.63GiB in this fixture. Therefore an inline packed-W2
loader alone does not remove the dominant partial traffic. Next bounded oracle:
four adjacent output columns per thread, vector loads/stores, unchanged Top6
FP32 addition order. Do not reuse old decode's5us reduction budget for this
large-prefill shape. Profiled timings are NOT new E2E scores or layer budgets.

Measurement caveats: profiler replaces the intercept queue with system memory
and does not preserve priority/CU masks. First two profiler attempts failed
before GPU counter collection: injected compiler version subprocesses mixed
SDKs; then rocminfo warning text broke AIter's architecture parser. The successful
probe leaves parent profiling intact, unsets injection only for actual child
`--version` probes, and sets GPU_ARCHS=gfx90a with an actual device-arch assertion.
No tool output or version string was synthesized. FetchSize+WriteSize collect
together; adding L2CacheHit exceeded the available same-pass counter grouping.
