# C16x32K wide-owner ABBA: +38.0224% over legacy MHC

Separate from16K acceptance. Actual input count is **524286**,
16 heterogeneous code requests,1M logical KV,32768 chunk,original V4/TP8/AR.
Frozen validated manifest: dsv4_c16_query_wide_20260915/inputs32k/prefill.json.

Only QUERY_OWNER_WIDE differs; full wide query16 enabled in both arms.
The real32K diagnostic passed2688 score-byte/logical-ID/physical-ID comparisons.
**Important:** batch1 legacy FP16-Fn fused-tail is common to both arms and
bypasses pre-mix owner. Capture timing-paths.json before quality/teacher tests;
do not claim the complete16K FP32 MHC chain is executing merely from flags.

run.py: A1 -> B1/B2 -> A2 fresh services (B legs share one process), three
scored waves per leg; four128-token quality waves and fixed64-token teacher
continuation per process. All service cleanup is owned and in finally.
summary.json/complete.json, not script presence, determine completion.
No global defaults or runtime kernels changed for this trial.

Completed:6163.794492 ->8507.415911 input tok/s,control drift-0.107192%.
All192 x128-token answers identical; A1/A2 and A1/B teacher continuations have
zero logprob delta and identical Top5 at1008 positions each. All services
stopped and8GCDs idle. This verifies indexer non-regression relative to the
legacy MHC path, NOT original FP32/20 MHC semantics or factual answer accuracy.
See acceptance.json and measured-legacy-launcher.sh; no default promotion.
