# DSV4 M2304/M2300 load-time prewarm checkpoint (2026-09-02)

## Change

The TP4/EP1/no-A2A production prefill profile uses two exact raw-FP4 MFMA64
specializations for the standard 4604-token request: M2304 followed by M2300.
Previously neither specialization nor the token-row MHC native module was
loaded by `_prewarm_mhc_kernels`; the first real request therefore paid the
JIT/module burst.

When all strict production guards are true, load-time prewarm now constructs
the cached JIT modules for:

- gate/up: E256, M2304/M2300, Top-6, local I512, K4096, G416, split-4, A64;
- down: E256, M2304/M2300, Top-6, N4096, local K512, G624, split-2, A64;
- native gfx90a token-row MHC post/pre.

The prewarm only calls module constructors.  It does not allocate routed
intermediates, launch the kernels, read weights, or change the forward
selector.  Decode/speculative profiles cannot enter the guard.

## Service validation

Physical GCDs 4--7, original checkpoint weights, TP4/EP1/no-A2A, queue-aware
4608 ceiling with a 2304 C1 budget:

- load-time prewarm: about 0.8 s/rank plus rank synchronization;
- first 4604-token real request: 2.869 s / 1604.6 input tok/s;
- the same cold request before explicit shape prewarm took more than 20 s;
- immediate warm rounds: 1.834 s / 2510.6 tok/s and 1.824 s / 2524.6 tok/s;
- later C1 samples showed 1.95--2.01 s jitter and are not treated as a kernel
  regression or a performance checkpoint.

The remaining roughly one-second first-request tail is outside these four MoE
modules.  A future exact startup warmup request should cover attention,
indexer, page metadata and the full scheduler path rather than extending this
module list blindly.

## Correctness and regressions

- France chat oracle: `The capital of France is Paris.`
- Native AR, 256 output tokens: 52.62 tok/s, `finish=length`, normal text.
- C32 real heterogeneous code prompts: 2533/2573/2548 aggregate input tok/s,
  median 2548.  This is below an earlier 2763 sample, but the candidate has no
  runtime branch or kernel change; retain it as observed service variance and
  require a later colocated ABBA before attributing any steady-path delta.

## Decision

Keep the guarded load-time prewarm.  It removes the dominant user-visible cold
JIT burst without changing steady inference.  Do not count it as a warm TTFT or
throughput improvement.
