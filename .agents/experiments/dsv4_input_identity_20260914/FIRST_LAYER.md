# First-layer closure: original V4 C16 prefill

2026-09-14 follow-up to `afe29cbb18`. Original V4 Flash, native TP8/EP1,
1M logical KV pool, chunk32768, same 16 distinct code requests and input-ID
verification. All candidates are default-off numerical diagnostics.

## Intervention sequence

1. `combined-l0`: fixed-order wo_a plus FP32 attention-output reduction.
   All 15 shared sampled rows' attention outputs and FFN inputs now match
   across A/B in all eight ranks. FFN output still differs (max 0.125).
2. `combined-ffn`: same mathematics, additional MoE stage tracing. Router
   logits, Top-K IDs/weights, routed outputs and final local FFN partials all
   match for the shared sampled rows. One rank's shared output differs by
   9.31e-10, but that difference vanishes in the combined partial. At the
   boundary row all eight input partials are identical, yet final BF16 FFN
   output differs across layouts. Versus FP64 sum/BF16 round, A has 2281
   differing elements (max0.125), B1694 (max0.0625).
3. `both-ar-l0`: also replace the FFN's final reduction with FP32 sum/BF16
   output. A1's local FFN partials match the preceding service in all eight
   ranks, confirming no expert-compute change or double reduction. Both
   batch layouts' FFN outputs match the FP64 reference on all sampled rows.
   Across A/B all 15 shared sampled rows' attention output, FFN input/output,
   MHC residual/post/comb match in all eight ranks.

Small internal Q and wo_b-partial differences remain in one rank each and
vanish before the replicated output. Thus this is closure of the **sampled
layer-0 boundary**, not every intermediate arithmetic value or the full model.

The same-order A1/A2 sampled layer-0 stages match for all three services.
The complete 128-token outputs remain non-identical:

| Run | A1/A2 exact requests | A1/B1 exact requests |
|---|---:|---:|
| combined-l0 |14/16|11/16|
| combined-ffn |15/16|12/16|
| both-ar-l0 |12/16|13/16|

These counts do not prove improved whole-model repeatability. Four inspected
both-ar output prefixes remain coherent code reviews; this is not an external
semantic oracle. Debug timings are not performance acceptance data.

## Exact scope

- `SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE`: original V4 BF16 wo_a, native
  TP8/EP1 eager M8192..36864, local group1, K4096/N1024. Fixed128 tile,
  eight waves, two stages. The service implementation passes100/100 real-input
  mutations on each of ranks5/6 and matches the offline candidate bitwise.
- `SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32`: existing attention diagnostic.
- `SGLANG_DSV4_DEBUG_PREFILL_FFN_AR_FP32`: original V4 FFN reduction only,
  same native large-M scope; excludes DP/CP/A2A, deferred RS and TP1 shared.
  Uses the existing skip-post-experts-reduction contract, then performs one
  FP32 process-group sum. No routed/shared arithmetic change.
- All three flags default OFF. Native decode, draft and target verification
  cannot enter these selectors. CK fixed-slot remains OFF in these runs.
- Each FP32 reduction temporarily needs512MiB atM32768. The service retains
  its1M pool; no persistent full-model weight expansion is added.

The MoE dump callback is owned by the original V4 eager debug scope and is
restored in finally. Other model callers have no callback. Parameter dumps
are no longer incorrectly sampled when their first dimension happens to equal
M; explicit weight-dump skipping makes multi-layer traces smaller.

## Collective alternative screen

RCCL exposes algorithm/protocol controls ([AMD reference](https://rocm.docs.amd.com/projects/rccl/en/docs-10.0.0/api-reference/env-variables.html)).
`rccl_order_oracle.py` freezes eight real FFN partial vectors and tests their
same-value shift across M32768/row24576 and M32767/row24575. Three input families:
original, alternating signs, per-rank power-of-two rescaling. All ranks agree;
five repeat outputs per case are exact. Timing includes two staging/output
copies for BF16 in this harness, unlike the earlier one-copy AR oracle.

| Mode, Simple protocol | BF16 rank-max median | Shifted BF16 elements |
|---|---:|---:|
| Ring, default channels |4.82–4.83ms|2203–2582|
| Tree, default16 channels |6.15–6.24ms|2383–2565|
| Tree, one channel |60.37–61.53ms|0|

FP32 output is shift-stable and matches FP64 reference in these fixtures;
Ring takes8.19–8.27ms, Tree9.86–9.92ms, one-channel Tree101–104ms.
Single-channel BF16 is stable but still differs from FP64 reference; precision
and invariance are different properties. No collective algorithm promoted.
Actual selected algorithm/protocol/channel counts are retained in RCCL logs.

## Remaining work

Expand sampled tracing across all43 layers, including MHC carry state, to find
the next propagating boundary. The default-off fixes are not a whole-model
determinism guarantee and have not passed performance ABBA. After numerical
attribution, return to C16 throughput optimization with real requests and
matched timing/quality criteria; no new prefill speed claim is made here.

## Completed43-layer follow-up

`all-layers` completed warmup/A1/B1/A2 with both FP32 reductions and stable
wo_a. Sample positions0/2047/8191, all eight ranks,43 layers, weight dumps OFF.
Only the final prefill group remains: A cases12/13/14/15 (M32768), B cases
12/13/2/15 (M32767). There are9 common sampled positions,72 rank-row pairs.
All16 requests' server input echoes still pass; GPU activation coverage is
explicitly only that last group and these sampled positions.

- Layer0 attention/FFN boundaries and MHC carry state match72/72 rank-rows.
- Layer1 sampled residual/norm still match. Q differs in two rank-rows at
  case15/position0 (ranks5/7). Attention core differs in six rank-rows at
  case15/position8191, where the sampled Q itself matches. This requires
  query/KV-history preparation checks; it does **not** isolate an attention
  kernel bug because unsampled histories are not proven equal.
- First replicated boundary difference: layer1 attention output,56/72 exact,
  max0.015625. It subsequently spreads through later layers and routing.
- A1/A2 sampled stages match across all43 layers. Full128 outputs match14/16;
  the two failures are cases4 and10, outside the retained final-group trace.
  Thus this cannot be used to blame the final head or decode from supposedly
  identical full hidden states. A1/B1 full outputs match10/16.

`all-layer-evidence.tar.gz` contains the summaries, request/response evidence
and logs, not the large activation dumps. `causal-vectors.json.gz` separately
publishes small BF16-bit-pattern activation vectors for independent checking
of the two collective boundaries; it contains no model weights.

Next: layer1 query/KV preparation, plus per-forward capture if attributing the
earlier groups' control drift. Preserve these coverage limits when interpreting
the data. First-layer fixes remain defaultOFF, and no production speed result
is claimed for them.
