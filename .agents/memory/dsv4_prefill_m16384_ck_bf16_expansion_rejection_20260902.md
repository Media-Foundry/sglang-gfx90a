# DSV4 TP4 M16384 CK BF16 expansion oracle (rejected)

Date: 2026-09-02

## Scope

This experiment tested a variable-M CK/AIter grouped MoE path for the TP4,
EP1, no-A2A DeepSeek-V4-Flash prefill path. The checkpoint weights remained
the original packed FP4 tensors. The oracle expanded the current layer's
routed weights to BF16 at runtime and then called the generic CK two-stage
grouped MoE with the exact DSV4 bounded-SwiGLU activation.

The production selector was default-off and was removed after the service A/B.
The standalone kernels and benchmark scripts are retained as diagnostic
oracles only.

## Hardware and toolchain

- Physical GCDs 4,5,6,7 for TP4 service; physical GCD 4 for standalone tests.
- `amd-smi process --general --sort-by-pid` was run before GPU experiments.
- AIter compilation must use `/opt/rocm/bin/hipcc`; the Conda `hipcc` misses
  `thrust/complex.h` for this CK build.
- Working launch environment used `ROCM_HOME=/opt/rocm`,
  `HIPCC=/opt/rocm/bin/hipcc`, `CXX=/opt/rocm/bin/hipcc`, and put
  `/opt/rocm/bin` first in `PATH`.

## Standalone evidence

The generic CK variable-M BF16 grouped MoE, excluding FP4 dequantization,
measured approximately:

- M4608 skewed routing: 6.507 ms median.
- M16384 Zipf-skewed routing, max occupancy 14940: 14.168 ms median.

The simple full-weight FP4+E8M0 to BF16 expansion cost was approximately
4.35 ms for both routed matrices. A synthetic balanced full stage was
11.505 ms at M4608/P128 and 20.959 ms at M16384/P384.

## Real routing observation

Real M16384 prefill over 32 heterogeneous code prompts was highly skewed.
Learned-router layers included experts with 9708, 11442, 13526, and 13848
assignments. Fixed P512/P768 padding is therefore invalid for this workload;
the next kernel must support variable M per expert or occupancy-bucketed work.

## Service result

Correctness screen:

- France prompt returned `Paris.`.

C32 screen used 32 real heterogeneous code prompts, 73724 input tokens total,
one generated token per request:

- wall time: 102.9391 s
- aggregate input throughput: 716.19 tok/s
- all 32 requests completed

The accepted production baseline is approximately 2.75k input tok/s on the
same request manifest, so this oracle regressed aggregate throughput by about
74%. It is rejected.

## Root cause and next constraint

The CK grouped GEMM itself is promising, but expanding every layer's complete
FP4 routed weights to BF16 on every forward dominates the service path. The
prototype also entered after the existing INT8 activation quantization and
sorter, then invoked CK's own routing preparation, duplicating fixed work.

Do not retry full online weight expansion. A viable successor must satisfy at
least one of these constraints:

1. CK/HIP consumes packed FP4 weights directly.
2. Only active expert tiles are unpacked, with reuse across all assignments.
3. A bounded persistent cache stores only demonstrably hot expert tiles.

The selector must also occur before the old INT8 quantization/sorter so the
candidate does not pay both pipelines. Validate with France first, then the
same 32-request manifest and multi-round completion hashes.

## Startup note

The first TP4 launch spent about 227.5 s capturing BS1/2/4/8 decode graphs.
Four ranks each created 32 TorchInductor workers and contended on shared cache
locks. This is a cold-start/prewarm issue, not a GPU correctness failure.
Future work should precompile once into a persistent cache or serialize the
single-rank compilation phase before TP4 startup.
