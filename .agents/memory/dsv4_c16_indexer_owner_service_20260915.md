# C16 original-V4 query-owner indexer accepted:7.95k input tok/s

TP8/EP1/no-A2A/native AR, original weights,1M logical KV,32K prefill budget,
16 real heterogeneous8K code prompts/131069 input tokens. Full query projection
and compressor/cache updates remain replicated; only active query16 groups'
logits/Top-K are owned across8ranks. Exchange logical IDs via existing TP RCCL
group and reconstruct through local physical page tables. No attention/expert
approximation. CPU maps use already-available prefix/extend metadata once per
forward, no live GPU→CPU synchronization in timed path.

Fresh-service ABBA,3waves/leg:
A1=7373.83345;B1=7957.03975;B2=7950.87246;A2=7375.01017 input tok/s.
Centers7374.42181→7953.95610, **+7.85871%**. Raw wave-TTFT timestamps
recomputed,192 exact formal input echoes,zero KV hits. Warmups excluded.
All services stopped cleanly; all8GCDs free at closure.

Correctness: real8rank integrated helper variants pass; diagnostic fresh model
dual-computes full-local vs owner scores/logical IDs/physical IDs in same forward.
Its16x128-token answers equal the previous checkpoint. Formal quality96
input echoes:31/32 candidate answers equal current controls; remaining B.1
case8 equals historical paired-MHC experiment's A1/quality-1 control. Read all
candidate text branches: coherent/no repeated collapse within128tokens, not
proof generated code-bug claims are factual. Current B repeats15/16 versus
controls16/16, so do NOT claim global bitwise determinism or drift repaired.
The previously demonstrated same-input layer20→42 drift still needs first-op
localization.

Additional8rank helper tests: M8192/8193/12289/65536, mixed prefixes,
different physical page numbering, all-trivial untouched fallback. Oversmall
score width now rejected using CPU metadata. `tested_helper.py` preserves
exact measured body; AST test proves only the post-run safety guard differs.
14CPU tests pass including launcher profile defaults/explicit0 and existing
MHC scope. Native AR/spec/draft/CP/V4.1 excluded; width512..2048, group16,
runtime-M, ordinary EXTEND required. Wider history falls back, not truncated.

Promoted only in combined TP8 multi-request + prefill-throughput launcher:
`SGLANG_DSV4_C4_PREFILL_QUERY_OWNER=${...:-1}`. Direct-launch default remains0.
No C1/decode/DSpark performance claim. 1M capacity preserved in actual service.
Do not sum amd-smi per-process VRAM entries into per-GCD memory; that exploratory
sum exceeded physical memory and was rejected, not reported as a peak.

Evidence: `.agents/experiments/dsv4_c16_indexer_owner_service_20260915/`.
Latest service budget must now be reprofiled; the old1.405s logits +.137s
Top-K budget is pre-owner and cannot be counted again as remaining opportunity.
