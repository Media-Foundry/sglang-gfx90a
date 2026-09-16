# FP32 MFMA pre-mix split16: +5.00% TP8 C16 service, numerical alternative

Later same-day follow-up: `dsv4_premix_owner_20260916.md` records an independent
exact paired-kernel row-owner path at9975.30tok/s (+11.11%), with all tested
continuations and fixed-prefix logprobs unchanged. It is now the stronger measured
candidate. MFMA remains a separate default-off numerical alternative; do not add
the two improvements or treat the older8.96k exact-reference statement below as
the latest fastest result. Exact shared-supply screening is also now completed:
see `dsv4_premix_exact_supply_20260916.md`.

2026-09-16. Base b6b70bd5f9. Original V4 Flash, TP8/EP1/no-A2A, native AR,
original checkpoint/Fn precision,1M logical KV,32K chunk,C16 x approximately8K
diverse real source-code requests,zero prefix hits,131069 inputtokens/wave.

## Formal ABBA (three waves per leg)

| Leg | Median input tok/s |
|---|---:|
| A1 accepted exact pre-mix |8980.047185|
| B1 cooperative FP32 MFMA split16 |9428.705094|
| B2 same candidate process |9424.729969|
| A2 fresh exact control |8975.137251|

Control8977.592218; candidate9426.717531; **+5.002737%**. A1/A2 drift-0.054676%.
Warmup excluded. Four processes in total: diagnostic check, A1, B, A2. No
concurrent GPU microbenchmark. All process trees stopped afterward.

The accepted **exact-reference** checkpoint remains the approximately8.96k
configuration. The measured9.43k is an explicit **non-bit-exact numerical
alternative**, not silently the new production default.

## Integration and resource scope

- `SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA=1` enables the candidate; default0.
- Admitted only after existing residual/Fn/RMS shape,dtype,contiguity,device checks,
  under `mix_pair_active()`: original V4 native TP8 ordinary EXTEND with
  8192<=M<=65536. No AR,DSpark,draft,V4.1,TP4,CP/TBO or capture admission.
- Fixed split16,K64,vector global loading,padded LDS; FnFP32,activationBF16.
  No target rows/experts skipped, no precision-reduced checkpoint.
- Per-call scratch48MiB atM32768,96MiB atM65536; no shared global workspace.
- All eight ranks selected the candidate inB. Both controls did not.
- Actual service logs show max_total_num_tokens=1048576 and chunk32768.
- Exact HIP post,corrected H16 peer attention,unique fixed-slot CK and vec4
  reducer are common toA/B; no changes to their arithmetic.

Three GPU integrated checks passed: M17 fallback exact; M8192/M32767 selected
runtime output bit-identical to the measured component oracle. Nineteen CPU
scope/negative-dispatch tests passed (one unrelated pytest asyncio config warning).

## Live arithmetic check, separated from timing

2720 eligible comparisons:85 boundaries x4 forwards x8 ranks. All entries
individually audited finite and within the stated1e-5 relative-L2 diagnostic
threshold; 340comparisons/rank. Maximum FP32 normalized-mix absolute delta
0.00048828125; maximum relative L2 3.0384976e-7. This threshold is a component
sanity gate, not a proof of whole-model quality or bit identity.

Diagnostic throughput6832.92 includes both reference and candidate plus
synchronizing comparisons, and is **not** a scored performance arm.

## Whole-model differences are reproducible and material to reporting

- France passes in all four processes.
-16requests x128tokens x4qualitywaves per timed arm: each arm is16/16exact on
  every repeat. Total192continuations across three timed processes.
- Cross-config continuations: only3/16identical;13differ. Requests6and8 already
  differ at the first outputtoken; other first-divergence positions are retained.
- Teacher forcing uses the same64-token suffix from the earlier accepted exact
  control, appended to each original prompt. Every input-ID echo is checked.
- The API returns a leading null logprob/top5 placeholder: exclude and explicitly
  validate it. Actual comparison covers16x63=**1008** positions, not1024.
- A1 vsA2:1008/1008top1same; selected-token logprob deltas exactlyzero.
- A1 vsB:982/1008top1same;26changed. Mean/max absolute selected-token logprob
  delta0.0271850053 /0.448553622. These are logprob differences, not raw-logit
  differences, and are not limited to floating-point last bits at model output.

Thus the new kernel is stable on the tested workload but changes the whole-model
trajectory. Tiny local mix errors can be amplified by the rest of the model;
this experiment does not localize that amplification to a particular router,
quantizer or later layer. Do not substitute output-hash mismatch for this
fixed-prefix evidence, or claim universal batch invariance.

All sixteen B excerpts were manually reviewed: coherent/relevant, no obvious
looping, but they are short and not scored against code-answer ground truth.
See quality-review.md. Keep candidate default-off pending broader long-output
quality evaluation; preserve exact-reference path and its speed label.

## Reproduction / evidence

`.agents/experiments/dsv4_mhc_mfma_service_20260916/` contains integrated.py,
service.py,analyze.py,summary.json,quality-review.md and raw service archive.
Service hashes actual sources before and after each phase; all three timing
plans match. Linux boot-ID/start-ticks ownership protects stop operations.
Only generated experiment files and intended kernel/helper/test changes belong
to this work; cuda_graph_runner_memory_usage.pickle remains unstaged runtime state.

Next: longer quality and sensitivity checks before enabling by default. If
exact-reference equivalence remains the required profile, investigate a
different exact pre-mix/data-reuse path; MFMA summation order cannot be called
legacy exact merely because Fn precision is unchanged.

One distinct untested exact candidate is cooperative activation staging across
several waves/column-pairs while retaining each wave's original K1024 products
and reduction tree. This would reduce repeated activation supply without using
MFMA summation. It requires matching actual scalar/DPP arithmetic and full
boundary measurement; neither feasibility nor a performance gain is established.
