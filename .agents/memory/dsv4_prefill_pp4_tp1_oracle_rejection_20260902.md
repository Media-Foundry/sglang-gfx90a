# DSV4 prefill PP4 / TP1 oracle rejection (2026-09-02)

## Objective

Test whether four pipeline stages on physical GCDs 4--7 can turn the existing
TP4 C32 prefill ceiling (~2.74k aggregate input tok/s on 32 distinct code
prompts) into a near-four-stage pipeline while retaining the original model
weights.

## Configuration

- `TP_SIZE=1`, `PP_SIZE=4`, `EP_SIZE=1`, no A2A
- `PP_MAX_MICRO_BATCH_SIZE=1`, `PP_ASYNC_BATCH_DEPTH=4`
- chunk 2304, 98,304-token pool, 0.96 static memory fraction
- token-row TP4 prefill optimization disabled
- original FP4/FP8 checkpoint; no weight conversion

Each PP stage loaded successfully.  Model memory was approximately 38--42 GiB
per GCD, so PP4 is not blocked by capacity.

## Blockers found

1. The TP4-specific raw-FP4 direct path correctly failed loud for the TP1
   expert shape.  Disabling it selected AIter's two-stage CK FP4 path.
2. AIter's bundled CK failed to JIT with clang 23 at
   `llvm.amdgcn.raw.buffer.load.lds`; clang 23 no longer accepts
   `-amdgpu-coerce-illegal-types=1`.  Temporarily selecting CK's existing
   `buffer_load_dword ... lds` inline-assembly implementation produced the
   module successfully.  The source toggle was restored after compilation.
3. The generated module still has no valid heuristic entry for the TP1 DSV4
   shapes. Decode graph capture failed at M=2.  With decode graphs disabled,
   the first real France request failed at M=16 with:

   `Unsupported kernel config for moe heuristic dispatch`

   Reported tensors were full-width TP1 experts (`I=2048` logical shape), not
   the tuned TP4 `I=512` path.

## Decision

PP4 currently cannot pass the mandatory France correctness oracle, so no
throughput claim is valid.  Making it testable requires a complete exact TP1
FP4 routed-expert dispatch/kernel configuration for both small-M decode and
large-M prefill.  This is materially larger than a scheduler-only oracle and
has no measured pipeline benefit yet.

Return to the validated TP4 path and prioritize the measured M2304/M4608 MoE
critical path (especially gate/SwiGLU-to-INT8 epilogue and intermediate HBM
traffic).  Revisit PP4 only after a standalone TP1 I=2048 routed-stage oracle
exists.

## Artifacts

- `/tmp/sglang_dsv4_pp4_oracle.log`: raw-direct contract rejection
- `/tmp/sglang_dsv4_pp4_oracle2.log`: LLVM intrinsic JIT failure
- `/tmp/sglang_dsv4_pp4_oracle3.log`: module built; M=2 heuristic rejection
- `/tmp/sglang_dsv4_pp4_prefill_oracle.log`: M=16 real-request rejection

