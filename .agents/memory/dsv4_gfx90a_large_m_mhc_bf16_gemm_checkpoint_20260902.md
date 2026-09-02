# DSV4 gfx90a large-M BF16 MHC GEMM checkpoint (2026-09-02)

## Configuration and contract

- Four physical gfx90a GCDs 4--7, original DeepSeek-V4-Flash weights.
- TP4/EP1/no-A2A, 83,968-token pool.
- C32 uses 32 distinct code-review prompts, 73,724 audited prompt tokens.
- `chunked_prefill_size=36864`, `prefill_max_requests=16` (16+16 admission).
- Corrected default-off large-M BF16-CK routed MoE: direct tiled
  FP4-to-preshuffled-BF16 expansion, DSV4 bounded SwiGLU, FP32 stage-2 atomic
  workspace.
- Token-row MHC ownership enabled.
- New `SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM=1` selector is prefill-only,
  requires M>=2048, and refuses CUDA graph capture.  Decode retains the native
  wave64 path.

## Component result

The existing wave64 kernel was decode-shaped and scales linearly at large M.
The candidate computes the 24-column projection with BF16 GEMM, then reuses
the native RMS-scale and scale-mix kernels.  The immutable FP32 Fn weight is
converted to BF16 once per layer and cached by device, data pointer, and tensor
version.

Physical GCD4 ABBA:

```text
M9216:  wave64 4.1089 ms, BF16 GEMM 1.2317 ms, 3.336x
M36864: wave64 16.3503 ms, BF16 GEMM 5.0180 ms, 3.258x
```

At M9216, 100 independent input/weight mutations passed with finite output and
cosine >=0.99999.  Representative maximum/mean absolute differences were
0.0212/0.00233 and cosine 0.9999973.  This is intentionally non-bitwise and is
reported as such.

## Service result

Corrected BF16-CK + token-row control (wave64 MHC):

```text
4175 / 4353 / 4548 input tok/s
median 4353
```

BF16 MHC GEMM candidate, five C32 rounds:

```text
4532 / 4667 / 4942 / 4986 / 4866 input tok/s
median 4865.5
trimmed mean 4825.0
```

The candidate improves the matched warm median by about 11.8%.  This is the
first corrected large-M service checkpoint; older 5.3k all-zero-output results
are not correctness evidence.

C1 4604-token warm prefill was 2.67--2.72k input tok/s after one cold/JIT
request, retaining the >=2.5k objective.  Native AR 256-token negative control
was 53.04/52.70/54.02 tok/s with identical hash `38c3d431e7c1dd65`.

France returned `The capital of France is Paris.`  A 32-request, 64-token
semantic run produced task-related code-review answers without NaN, garbage,
or repetition collapse.  Greedy hashes remain non-bitwise because both the
large-M BF16 routed path and BF16 MHC change FP32 association.

## Decision

Keep this as a default-off experimental checkpoint.  It clears the 5% service
gate and preserves C1/decode, but C32 remains well below the 10k objective.
The next work must target the remaining large-M dense projections, sparse
attention, collectives, and/or a more efficient packed-FP4 routed path.
