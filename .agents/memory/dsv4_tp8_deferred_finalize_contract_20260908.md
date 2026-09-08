# Call-owned deferred grouped-down finalize contract

Parent checkpoint bbbf6849f5. The preceding local fusion oracle saved ~1.60 us,
but production needs a way to carry partials across its existing shared join.

## Implemented, not selected by the service

`gfx90a_fp4_expert_down_grouped(..., defer_reduction=False)` remains Tensor-returning
by default, using the same compiled `.run()` as before. Explicit true is restricted
to E256/M32/T6/N4096/K256, no runtime-M/row-prefetch/logical-scale/zero-partial variant.
It invokes the existing `.run_partial()` and returns a frozen per-call object holding
its partial and output. `finalize(shared)` runs the exact two-round local fusion.
No context variable, global deferred buffer, monkey patch, or new server env exists.

The model/runner/dispatcher are NOT wired yet. Existing FlashInfer deferred finalize
is backend-specific; the standard FusedMoE forward also slices/contiguates its output,
so returning an opaque object through that path without explicit handling is unsafe.
The next step needs explicit dispatch/combine handling and native TP8/M32 guards,
not reuse of a hidden global tensor. C1, prefill and speculative selectors unchanged.

## GPU validation

Physical GPU4; AMD-SMI showed only baseline PID3325034 and its children before testing.
`check_dsv4_deferred_finalize.py` uses full production grouped-down kernels with
nonconstant packed FP4 weights and physical E8M0 scales, changing activations, scales,
router weights, and remapped expert IDs. This validates interface/implementation parity,
not an independent raw checkpoint dequantization oracle.

- Default vs deferred: 100/100 BF16 int16 bit-pattern exact, finite baseline.
- Two outstanding calls have different partial addresses; changing inputs and finalizing
  in reverse order gives both original default-path outputs exactly.
- 1000 single-stream graph replays exact; allocated memory stable.
- Existing fork-before-down/join-after-shared pattern: 100/100 bit-pattern exact;
  1000 graph replays exact and allocated memory stable.
- Shared producer in that test is BF16 multiply, NOT actual shared expert GEMM.
- No E2E throughput, model semantic validation, or cross-rank collective result yet.

Partial size remains 3,145,728 bytes per call. Extended lifetime across the shared
join can change graph allocator reuse; retained 1M KV capacity and actual graph pool
delta must be measured on integration. Do not claim zero added service VRAM from
the component allocation-stability check.

Baseline service and its 1,048,576-token pool were not restarted or reconfigured.
Raw results: companion JSON. No new production speed checkpoint.
