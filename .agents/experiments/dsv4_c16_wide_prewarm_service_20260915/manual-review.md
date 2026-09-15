# Bounded long-input output review

64 quality responses,128 tokens each.55 match previously reviewed identical
input responses from the prior16K/32K wide-query service trials. Read all nine
new responses:16K wave1 case3;32K wave0 cases8/14;32K wave1 cases2/5/6/10/13/14.
No obvious garbling or repetition collapse in these excerpts. They address
allocation, distributed helpers, override ordering, attention and expert routing.

This is NOT a factual correctness pass. In particular:

-32K case10 labels GFX95 "RDNA3" based on an import of is_gfx95_supported.
  That hardware-family assertion is not established by the quoted code.
-32K case13 calls sequential creation of device and CPU groups a critical
  ordering issue; sequential new_group calls alone do not establish deadlock.
  The128-token excerpt ends before a complete supporting argument.
-32K case2 asserts premature qkv_a deletion but the saved128-token excerpt
  does not yet demonstrate the complete control flow or a valid repair.
-Approximate line references and remaining claimed defects were not certified.

Same-input full-output repeats are14/16 at16K and7/16 at32K. Existing trials
also had unstable controls; this is not a same-run A/B against stage2 or against
unprimed execution. Therefore no causal attribution of output drift to prewarm,
stage1, wide-query, admission or MHC is made. Matching bytes of compiled
artifacts proves cache/dispatch alignment, not global model determinism.

Follow up with the actual long-input per-call stage1/stage2 equality checker
before attributing lower repeatability to the attention pipeline. Keep exact
input IDs, physical metadata and existing numeric-path labels distinct.
