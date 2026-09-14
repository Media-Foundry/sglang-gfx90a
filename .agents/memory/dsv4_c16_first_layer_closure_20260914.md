# C16: sampled layer0 boundary now stable, whole model still drifts

Follow-up to `afe29cbb18`; see
`../experiments/dsv4_input_identity_20260914/FIRST_LAYER.md` for exact runs.

Inputs are verified at client, server echo and GPU, keyed by request ID and
absolute token position. TP8/EP1 native,1M KV,32768 chunk,16 real code requests.

Stable wo_a + FP32 attention AR closes sampled attention output and FFN input.
The next difference is FFN's final BF16 reduction: all eight local partials
match across layout changes, yet global output differs (max0.125). Router,
Top-K and routed compute are not the cause in these sampled layer0 rows.
Adding FP32 FFN AR makes sampled attention output, FFN input/output and MHC
residual/post/comb match15/15 common rows on all eight ranks. The local partials
are unchanged by the replacement, confirming no double reduction.

Whole128 outputs remain non-identical (both-AR control12/16, permutation13/16).
Do not declare model determinism or accuracy recovery from one layer.
All-layer sampling is the next step; include MHC carry state, not only hidden.

New `SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE` and
`SGLANG_DSV4_DEBUG_PREFILL_FFN_AR_FP32` remain OFF. They exclude decode/draft/
verification and non-TP8/EP1 profiles. CK fixed-slot stillOFF. No speed accepted.

RCCL Tree/default channels still drifts and loses to Ring. One-channel Tree
eliminates fixture shift drift but costs60–62ms versus Ring4.82–4.83ms. Reject
for performance. These timings include two copies; do not compare mechanically
to the earlier one-copy4.4ms oracle. Numerical precision and order invariance
are distinct: single-channel BF16 still differs from FP64 reference.

43-layer follow-up completed (`all-layers`): positions0/2047/8191, eight ranks,
retained final prefill group only. Layer0 boundary/carry states72/72 exact;
next propagated difference is layer1 attention output56/72 exact, max0.015625.
Its sampled norm inputs are identical. Q differs at case15/pos0 on ranks5/7;
attention core differs at case15/pos8191 on six ranks despite equal sampled Q.
Unsampled KV histories are not verified: do not label this an attention-kernel
bug yet. A1/A2 sampled states match across all43 layers, but control output
failures are cases4/10, not in the retained last-group GPU trace. Full outputs
14/16 same-order and10/16 cross-order. Next investigate layer1 prepare and/or
capture all prefill forwards to cover those earlier requests. No E2E speed
acceptance and no production default changes.
