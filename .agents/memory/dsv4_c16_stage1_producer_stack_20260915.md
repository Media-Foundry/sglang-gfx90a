# Stacking optional owner-Q producer on accepted stages1 attention

Tested commit0de5b284f3. No production source/default changes this turn.
Original DeepSeek-V4-Flash, TP8/EP1/no-A2A/native AR, original weights,
1,048,576 logical KV,32768 chunk,16 heterogeneous8K code requests,
131069 actual input tokens/wave. This is prefill, not decode throughput.

Both arms fix `SGLANG_DSV4_PREFILL_ATTN_STAGE1=1` and accepted query-owner/
query16/runtime-M/MHC flags. Only `SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER` differs.
All intrusive diagnostics disabled. One CPU flag-isolation test passed;
driver/analyzer compile checks passed. Prerequisites include prior full-Q
same-BLAS oracle, fixed-continuation numerical evaluation and accepted
stages1 attention service checks. This trial does not repeat that teacher test.

## Formal A1/B1/B2/A2

Three fresh services, three waves per leg, excluded warmup per service.
Rates recomputed from raw independent streaming client timestamps as131069
divided by earliest request begin to latest first-token arrival. All192 formal
input echoes exact, zero cache hits, one output token per request. Source and
input manifests agree across arms; no measured source changed afterward.

| Leg | Three input tok/s measurements | Median input tok/s | Median wave seconds |
|---|---|---:|---:|
| A1 |8391.175 /8389.349 /8375.992|8389.349038|15.623262|
| B1 |8701.520 /8694.589 /8684.320|8694.588885|15.074778|
| B2 |8690.295 /8687.776 /8699.843|8690.295310|15.082226|
| A2 |8391.073 /8379.270 /8393.283|8391.073452|15.620052|

Mean of leg medians **8390.211245 ->8692.442097 input tok/s (+3.602184%)**.
Return control reproduces. Request TTFT medians A1/B1/B2/A2:
9.955995/9.595037/9.589015/9.955440seconds (not wave duration).
No compile markers in formal log windows. All eight ranks hit stages1 in
both arms, owner/producer respectively withchecks0. Each leg's observed
full/local/width signatures:32766/3072/2048,32767/3072/2048,32768/3072/2048.
This verifies observed sizes, not every internal token-row ordering.

## Quality and drift remain separately reported

France passed before any large-prefill owner/producer/stages1 hit in each
service. Two128-token quality waves per service:96 exact input echoes,
zero cache, output lengths and tokenizer decoding checked.

Full128-token repeats: A1=15/16, B=15/16, A2=16/16. Independent controls16/16;
A1.0 versus B.0 only3/16.31/32 candidate excerpts match previously reviewed
producer responses. The remaining case8 opening is coherent and topical;
inspection of the ACTUAL input confirms the named distributed helper and
MIN/cpu_group operation. See manual-review.md for limits. Do not claim every
generated bug diagnosis is correct or that no long-generation risk remains.

The fixed-continuation producer record remains relevant:
`dsv4_c16_owner_producer_teacher_20260915.md` found cross-path drift above
control noise, including non-small-margin top1 flips. Its exact numerical
statistics are not re-measured or newly asserted for this stacked service.
Same original checkpoint precision does not imply same downstream logits.
No evidence here repairs the separate CK atomic reduction nondeterminism.

## Disposition

Keep default accepted production **8384.698465 input tok/s** (stage1,
producerOFF). Stacked **8692.442097** is a qualified **opt-in** result with
the documented producer numerical tradeoff; producer remains default-off.
The increments have actually been tested together now, rather than adding
independent percentages. No decode/spec/draft/V4.1 scope expanded.

All three owned services stopped cleanly; amd-smi reported no running process
on any of8GCDs. Original user pickle/unrelated dirty files preserved.
Artifacts `.agents/experiments/dsv4_c16_stage1_producer_stack_20260915/`:
driver/sweep/analysis/review/test, bounded review, and evidence archive with
raw manifests, timings, responses, logs, source hashes and stop records.
Archive hash is in evidence.sha256.

## Coverage inventory for next work

Do not rebuild long-context fixtures or call those workloads untested.
Existing real16K/32K and mixed-prefix tests are recorded in
`dsv4_c16_query_wide16k_service_20260915.md`,
`dsv4_c16_query_wide32k_service_20260915.md`, and
`dsv4_c16_mixed_prefix_service_20260915.md`.
They predate recent owner/stages1 changes. Wide-query had warm benefits but
remains default-off pending cold-width closure. Mixed-prefix rates count
328190 newly computed tokens separately from524286 full input tokens, excluding
priming; do not compare full-input rate to zero-prefix8K rate.

Next useful work is updated exact-stage1 coverage using those existing inputs
and investigating wide-width cold compilation, not repeating query4/empty-tile
or treating the old6.88k/2.76s-attention budgets as current. Goal stays active.
