# C16 drift: input identity established, two numerical contributors

See `../experiments/dsv4_input_identity_20260914/README.md` and evidence archive.

Original V4 Flash, native TP8/EP1, 1M logical KV, chunk32768, CK fixed-slot OFF.
Shared requests have identical outbound, server-echoed and GPU-traced input IDs
and positions. Responses are keyed by explicit request ID; cache hits are zero.
Changing batch composition moves one identical request's first row by one.

All-rank tracing matters: rank0 input through wo_b_partial is exact, but rank5/6
wo_a already differs slightly. Independently, identical eight-rank real partials
in M32768/row24576 versus M32767/row24575 produce 2447/4096 changed outputs with
BF16 RCCL. FP32 RCCL and AIter float-accumulation paths eliminate the difference
on this fixture. This is not proof of universal FP32 or model determinism.

The 64-MiB default AIter capacity does not admit these ~256-MiB messages. Enlarged
AIter screen improves this fixture's numerical stability but is slower: legacy
16CTA ~5.43–5.46ms versus RCCL BF16 ~4.39–4.52ms, including staging. 8/32/48/64/80
CTAs are worse. No production collective change accepted and no E2E speed claim.

Next: test small-message chunking and large-M row-stable wo_a. Keep checkpoints,
AR/decode semantics and KV capacity. Do not conflate input identity with batch
identity, or this communication contribution with CK atomic-stage2 drift.

Follow-up: AIter chunking 1024..8192 rows also loses (~5.72ms best). A new
default-OFF `SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32` diagnostic was tested in a
fresh C16 service with 1M KV. It replaces only large native TP8 attention-output
reduction, never decode/draft/expert compute. Eight ranks confirm selection.
The measured boundary's collective output matches FP64 sum/BF16 rounding;
cross-order differences fall 2447 -> 4 elements. Residual differences follow
rank5/6 wo_a partials. Full128 output repeats14/16, not fixed globally.

Real-weight wo_a oracle reproduces baseline exactly. Moving one input row
changes one element on rank5/6. M-padding, F.linear and disabling reduced BF16
precision do not fix it; transposed-output GEMM and fixed-K64 do, but cost ~2x
and ~2.6x respectively. No speed claim or production promotion. Runtime input
identity is now verified, not merely inferred from a reused request manifest.

The fixed-order wo_a candidate was subsequently tuned to BM128/BN128/BK128,
8 waves/2 stages: 2.41–2.46ms, versus library 2.02–2.11ms. Each of ranks5/6
passes 100/100 shifted-row mutation tests over the 20 sampled real inputs.
FP64-reference max BF16 error 0.001953125. It remains an offline experiment,
not a service selector. Next useful step is combined first-divergence validation
with FP32 attention AR; performance defaults have not been changed.
