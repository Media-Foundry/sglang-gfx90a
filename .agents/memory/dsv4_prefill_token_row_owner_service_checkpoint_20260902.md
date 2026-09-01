# DSV4 TP4 token-row-owner prefill service checkpoint (2026-09-02)

## Scope

- DeepSeek-V4-Flash original checkpoint
- 4 physical gfx90a GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- TP4 / EP1 / no A2A / native AR
- chunked prefill size 2304
- experimental selector: `SGLANG_DSV4_GFX90A_TOKEN_ROW_MHC_PREFILL=1`
- selector is default-off and requires non-speculative extend, M>=2048,
  M divisible by four, TP4/EP1, no CP/TBO/A2A, and AIter custom collectives.

## Dataflow

The production experiment keeps full hidden rows token-owned during MHC while
leaving attention and MoE tensor parallel:

1. attention/MoE emits its rank-local TP partial without the trailing AR;
2. AIter row reduce-scatter produces M/4 complete hidden rows per rank;
3. gfx90a native MHC post+pre and RMSNorm run only on owner-local rows;
4. AIter all-gather publishes normalized full rows for the next projection;
5. residual/post/comb remain owner-local across all 43 layers.

The TP4/EP1 shared expert is TP-sharded in this profile.  The existing MoE
path first forms the BF16 rank-local `routed_partial + shared_partial`, then the
token-row path reduce-scatters that combined tensor.  This preserves the
reference local-add/reduction order.  Replicated TP1 shared experts fail loud.

An RCCL barrier follows the final eager all-gather.  AIter's unregistered
all-gather does not provide a rank-wide completion fence; without this barrier
a fast rank may enter the following registered decode graph while a peer still
retires the prefill epoch and both reuse communicator flags.

## Correctness

- Standalone two-boundary oracle: eager and 1000 graph replays bitwise exact.
- Chained owner-local residual/post/comb state: bitwise exact.
- Five warm 4604-token service requests after JIT produced one stable first
  token hash in the actual hit run.
- `long -> short France -> decode` returned `The capital of France is Paris.`
- `long -> short Python-code -> long` completed with HTTP 200; code response
  was semantically correct.
- Warm short native AR repeated 64-token hash `fb88908cc7287b21` twice.
- Default-off selector leaves decode and speculative paths unreachable.

## Performance

Control (token-row selector disabled), warm 4604-token TTFT:

```text
2.116, 1.870, 1.895, 1.889 s
steady median ~= 1.895 s
```

Actual selector hit after native-module warmup:

```text
1.680, 1.790, 1.719, 1.786, 1.803 s
median = 1.786 s
```

This is about 5.8% lower median TTFT.  Input throughput moves from roughly
2.43k to 2.58k input tok/s for the 4604-token probe, clearing the 2.5k C1 goal.
The native HIP module adds a cold/JIT tax (first hit observed at 8.08 s), so it
must be explicitly prewarmed before production use.

With the final completion fence, a repeat `long -> short -> long` run measured
1.698 s and 1.715 s on the two long requests.  The first short shape still has
its own AIter GEMM JIT tax; subsequent 64-token native AR was 49.09/49.31 tok/s
in this generic TP4 service profile.

## Rejected/clarified observations

- Earlier apparent candidate results around 1.85 s did not hit this path:
  `self.use_fused_mhc_post_pre` was false because ROCm server resolution
  disables TileLang MHC unless explicitly requested.  Those runs are controls.
- The new path therefore calls the validated gfx90a native MHC boundary
  directly and does not enable TileLang globally.
- The previously tested MFMA64 gate->INT8 I32-owner epilogue remains rejected:
  complete routed stage regressed 24.8% at M2048 and 27.3% at M2304.

## Next work

- Add startup prewarm for M2304/M2300 token-row MHC to remove the cold 8 s hit.
- Measure 1K/4.6K/8K/16K/32K and prefix-cache ratios; M<2048 currently stays on
  the control path.
- Measure real heterogeneous C4/C8/C16/C32 prefill and colocated decode before
  enabling this selector by default.
- Replace the host-visible RCCL completion barrier only if a graph-safe device
  epoch fence proves equal correctness and lower E2E latency.
