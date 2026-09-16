# Read-only follow-up audit during frozen32K indexer ABBA

Do not edit runtime until A2 has stopped and current source hashes are checked.

Confirmed in actual B process2531860 environment: FP16_MHC_DOT=1,
MHC_SINKHORN_ITERS=8,PREFILL_MHC_CONFIG_ITERS=0,
NATIVE_MHC_POST_PRE_FULL=0,NATIVE_MHC_POST_PRE=0.
BF16_MHC_DOT absent; its EnvBool default isFalse.

Source contracts in `kernels/ops/layernorm/mhc.py`:

- `_mhc_fusion_admitted(global_batch_size)` returnsTrue unconditionally for1.
- fused post/pre calls `gfx90a_mhc_splitk_fused_tail_triton` before trying
  paired pre-mix; it reads FP16Fn if requested and uses env Sinkhorn iterations.
- Merely disabling fused-tail is insufficient: the later split-K pre-mix
  still takes batch1, and the later native-finish/weighted-RMS branches also
  have single-request priority.
- `hc_split_sinkhorn` uses env iterations for hint1, but20 otherwise. Thus
  merely forcing FnFP32 does NOT restore the multi-request reference.
- Initial model `hc_pre` passes no batch hint to hc_split_sinkhorn, so its
  ordinary first boundary already uses20. The85 fused boundaries differ.

Proposed separate experiment, NOT implemented or claimed tested:

1. Opt-in original-V4/TP8/native eager large-prefill common arithmetic policy,
   gated by existing mix-pair scope and globalM8192..65536. Small decode,
   DSpark/draft, CP/TBO, othermodels and TP4 untouched.
2. Within MHC only, use an explicit no-singleton-dispatch hint for the selected
   prefill boundaries. Do not mutate ForwardBatch.batch_size or invent a
   different scheduler batch. Route all singleton priorities consistently,
   including full-native, splitK, fused-tail, native-finish and weighted-RMS.
3. Preserve original FP32Fn, existing exact HIP post, paired/owner pre-mix,
  20-iteration Sinkhorn and current weighted sum/RMS reference. Reject
   conflicting BF16-dot/approx-iteration overrides rather than silently mix.
4. On identical live inputs compare batch1 candidate to existing batch2/None
   FP32 reference, bytewise all four output tensors. Compare legacy FP16/8
   separately and disclose expected differences; do not demand equality to a
   different-precision/iteration baseline or call it arbitrary global bit-exact.
5. Then fixed-continuation/full-output and service tests, with decode regression.

This audits one explicit source of batch-dependent behavior; GEMM shape and
other rounding differences are separate. Do not claim this alone fixes every
historical model drift. Existing16K accepted trial remains valid for its actual
multi-request path; current32K indexer trial is only relative to legacy FP16/8.
