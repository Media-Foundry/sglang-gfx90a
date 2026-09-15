# Default-off owner-Q producer integration and fresh quality gate

Active goal turn is progress: real service integration and full-chain checks
completed. Accepted speed still7953.956103 input tok/s; no new performance
measurement or default promotion. Original V4/TP8/EP1/native AR, original
checkpoint,1M logical KV and32K prefill budget retained in both fresh services.

## Implementation

`SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER=1` admits a separate early branch before
normal full-Q preparation. It requires the accepted owner/reuse flags and all
original-V4 TP8 ordinary-EXTEND scope guards: no AR, spec/draft/MTP, V4.1,
CP/DCP/PP, TBO, rewritten batches, prefill graph, external indices, HiSparse,
TopK-V2 or unsupported indexer backends. Default is0; launcher not changed.

`gfx90a_indexer_producer.py` keeps full-M metadata and the full weights_proj
(including precomputed-weight reuse). Only owned q_lora/positions/weights
rows enter wq_b and RoPE/Hadamard/quant. Full compressor/index-key writes
remain on full x. Selected logical IDs use the previously validated integer
TP gather and local physical-page reconstruction. No zero-filled full Q
placeholder and no treating compact-Q M as the metadata M.

Unsupported host metadata/layout returns False BEFORE query/cache computation
and collective admission. After admission, broken contracts fail rather than
silently taking a rank-dependent fallback. Width512..2048 and M8192..65536
guards remain; wider live contexts are not truncated to fit.

`gfx90a_rocblas_linear.py` owns a separate rocBLAS handle per backend/runner,
host thread, device and retained live stream (not per model layer). Weakref
finalizer releases the handle. BF16 operands/output, FP32 accumulation,
no new weight representation and NO process-global BLAS switch. This is
still experimental machine-specific ROCm integration, not general upstream
backend support. Captured decode graphs cannot enter it.

## Validation before serving

22 CPU tests pass, including actual selector truth-table checks for mode,
parallelism, original-model marker, all dependent flags, default-off behavior,
and host width/prefix/M bounds. An old test initially selected the new early
owner guard rather than its intended old diagnostic hook; its selector was
corrected to distinguish the two paths (1 failed/21 passed ->22 passed).
Existing pytest asyncio_mode warning remains.

The actual production Linear class (not just its experimental predecessor)
passed real-input M3/17/3072/32767 comparisons against PyTorch rocBLAS, alternate
ordered-stream checks,100 graph replays and20 fixed-pointer input mutations.
Evidence: `dsv4_c16_owner_producer_20260915/production-rocblas.json`.

## Fresh candidate with full-chain checking

`.agents/experiments/dsv4_c16_owner_producer_service_20260915/run.py` ran
`B-check` with `SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK=1`, then `A-quality`
with producer disabled. Same16 real8K code prompts/131069 input tokens,
two256-output-token waves per service, greedy/ignore_eos, zero prefix hits.

All64 input echoes exact, completion length and decode-to-text checked.
Both France checks returned Paris BEFORE any large-prefill selector hit.
Both services stopped cleanly; source hashes frozen throughout each process.

Candidate logs establish all8ranks hit M32767 and M32768, local3072, width2048.
Every admitted checked call compares compact Q/folded weights/scores and
full-row logical/physical selection to a FULL-Q reference using the SAME
explicit rocBLAS backend and same original weights projection. All checks
passed throughout both full-model prefill/generation waves. This is a wider
gate than the previous one-layer oracle, but is NOT equality with old
hipBLASLt production and does not cover every possible shape/length.

## Output differences are real: do not relabel them as unchanged quality

- Control A0 vs A1:16/16 whole256-token outputs exact.
- Candidate B0 vs B1:14/16 exact; cases4 and15 diverge after196 and129 tokens.
- Each candidate wave vs A0:only1/16 whole outputs exact.
- Cross-config common prefixes per case:
  237,157,33,46,39,48,63,10,106,256,91,3,0,32,50,22.

All candidate wave0 answers and both changed wave1 branches were read. No
repetition/collapse observed within these256 tokens; responses remain topical
code analysis. But they sometimes choose different purported defects, not
merely different wording. For example case2 changes from a sink-sharding
hazard to an uninitialized-sink hazard; case7 changes from multimodal merge
analysis to output_ids mutation invariants. Both control and candidate also
contain questionable collective-ordering explanations in case13. None of
these generated bug claims has been independently validated as fact.

Thus the service routing/ABI/capacity and same-backend correctness gate passes,
but equivalence of answer quality or old target logits is NOT established.
Do not blame all remaining cross-config/in-arm drift on CK atomics without
new first-divergence evidence. Existing CK atomic drift remains a separate
reproduced issue. Keep producer DEFAULT OFF.

Next: fixed-continuation target logprob/logit comparison before performance
promotion. Then noninstrumented service ABBA if numerical/quality acceptance
is justified. This diagnostic candidate ran extra full-Q work and synchronizing
assertions, so its wall time MUST NOT be reported as the candidate speed.

For teacher forcing, do not inadvertently lengthen8K prompts beyond the width
2048 admission guard and then compare two fallback paths. Preserve actual
producer hits and identical token IDs/layout in that test.

Evidence archive `producer-integration-quality-evidence.tar.gz`:2364390 bytes,
SHA256 `73ffe4576e51e85a2f1f8e1cc7944b79b07f383255d338a64f3d3fb925b8fece`.
Includes both fresh services' complete request/response manifests, config,
source hashes/patches, logs, owned-stop proof and comparison/manual review.
Final GPU check reports no running processes on all eight GCDs.
