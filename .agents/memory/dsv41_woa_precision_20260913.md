# V4.1 first-decode precision investigation, 2026-09-13

## Scope

Continuation of `dsv41_resume_prefix_20260913.md`, not a performance trial.
Original V4.1 weights, TP8/EP1/no A2A, eager native AR, chunk2304, 32768
pool/context, pinned host Engram. Initial HEAD `1f2389d520`.
Unrelated dirty `cuda_graph_runner_memory_usage.pickle` is preserved/excluded.

Fixed event-sessionization SQL input: 203 IDs from the committed
`dsv41_resume_prefix_20260913/events-prepared.json`. Generate exactly two
tokens, then fresh-cache recompute those same 204 prefix IDs. No independently
sampled second trajectory. Actual first two IDs are `[666, 31151]`.

## Evidence chain (before repair)

Three fresh diagnostic processes reproduced the same first-decode difference:
committed ID 31151 agrees, but max common Top20 logprob difference is
1.2500200271606445. Two identical recomputes agree. Hooks copy to CPU and
therefore invalidate any timing/absence-of-race claim.

1. Full module traces aligned cached row0 at absolute position203 with
   prefill row203 (M204), not padded bookkeeping rows. Embedding, initial
   norms/projections, layer0 MoE/Engram all agree. Layer1 gate has a small
   non-selected-logit difference, but Top6 IDs/weights and MoE output agree.
2. Layer4 attention output first differs in one BF16 element (max 0.00048828125),
   which is lost at the next HC collapse. Layer6 attention output differs in
   549 elements (max 0.0078125); error then propagates to later norms/MoE.
   Comparing the *historical* first203 rows of M203 vs M204 full recompute
   shows exact agreement through the inspected first eight layers.
3. Layer4/6 rank0 Q, positions, selected 128 SWA records and 102 compressed
   records are byte-identical between cached/recompute. Selected packed KV
   capture respects page-planar layout: all data576 records, then all scale8
   records. Naively reshaping to per-token584 records is NOT valid.
4. Full eight-rank wo_b replay uses actual runtime weights/scales and hidden
   inputs. Both observed collective outputs equal the BF64-summed local
   partials rounded once to BF16. Do not blame collective ordering here.
5. Crucial layer6 rank1: **one** wo_b input element differs by 0.00048828125.
   FP8 activation quantization makes its local wo_b outputs differ in 1743
   elements (max 0.001953125); this explains the 549-element global difference.
6. Replayed actual attention→inverse RoPE→wo_a reproduces all16 captured
   wo_a outputs (layers4/6 × eight ranks) for both decode and prefill. Inputs
   to attention agree; pure attention M1 vs M204 agrees. The relevant
   difference is wo_a's GEMM row placement/reduction, not cache addressing.

## Important oracle pitfall

The first component version made local heads contiguous and compared row0
of a repeated M204 tensor. This did not reproduce the actual service path.
The corrected oracle preserves the 64-head output's stride while viewing
the eight local heads, and compares row203, not row0. It requires matching
the actual recorded wo_b inputs on both sides. All16 cases reproduce exactly.
Older `/tmp/*woa-contract*` / `*strided-contract*` reports without this
placement/reproduction check are exploratory, not the acceptance evidence.

## Narrow candidate

`wo_a_bf16_invariant` reuses the upstream batch-invariant persistent BMM
kernel with fixed M16/N64/K64, four waves, one stage. It keeps FP32
accumulation and one BF16 output rounding, regardless of decode/prefill M
or row placement. It adds no weight cache or global batch-invariant mode.

Only `deepseek_v41.py` consumes the new opt-in
`SGLANG_DSV41_WO_A_INVARIANT`; default is **False** while acceptance is in
progress. Existing V4 Flash code is untouched. Validated operand contract:
HIP, BF16, G1/G2, R1024, D4096, contiguous weights and unit K input stride.

- Original service-reproducing wo_a: 7/16 cases have M/row-placement drift.
- Fixed reduction: 16/16 agree, including every repeated row in M204.
- Against FP64→BF16 reference: max relative L2 0.000224343; this is not a
  claim of correctly rounded FP64 equivalence for every element.
- Independent mixed-row checks: 66 cases, M1/2/3/15/16/17/32/64/128/203/204/256,
  G1/G2, non-contiguous local-head storage, first/middle/last row alignment.
- 100/100 input mutations and 1000/1000 individually checked HIP Graph
  replays agree. Only this component is graph-validated; serving is eager.
- CPU suite at candidate startup: 121 tests +16 subtests passed.

## Reproduction / artifacts

Committed reports and launch scripts:
`.agents/experiments/dsv41_first_decode_precision_20260913/`.

Full raw tensors (including runtime weights) stay on this machine, not Git:

- `/tmp/dsv41-first-decode-precision-20260913/`
- `/tmp/dsv41-attention-precision-20260913/`
- `/tmp/dsv41-rank-attention-precision-20260913/`

Tools: `compare_dsv41_row_trace.py`, `check_dsv41_wob_rank_contract.py`,
`check_dsv41_attention_batch_contract.py`, `check_dsv41_woa_invariant.py`.
Use the DS conda Python and launcher-equivalent source/AOT PYTHONPATH.
GPU component checks used physical GPU4 only; amd-smi was checked before
GPU work. Do not derive service throughput from these diagnostic runs.

Full-model candidate prefix/semantic acceptance is pending below; component
success alone does not establish whole-model numerical correctness.

## First candidate E2E: not accepted

New process ready16:06:21. Same two-token rollout and fixed204 recompute:
IDs still `[666,31151]`, but max common logprob difference **1.499851**, vs
control1.250020. First visible selected-row output divergence is now layer1
attention. Historical203 rows of full203/full204 still agree through the
first three inspected layers. Do **not** enable the candidate by default
or claim it fixes full-model precision. Fixing one M-dependent reduction
can expose a different upstream/downstream quantization boundary; the next
step captures layer1 Q/KV/projections on all eight ranks with the candidate.
