# V4.1 first-decode precision investigation, 2026-09-13

Status at handoff: **paused for the user's emergency job; whole-model
numerical agreement is not yet repaired**. Both experimental selectors stay
default-off. The final TP8 service is stopped and all eight GCDs are released.

Evening continuation after user authorization:
`dsv41_hc_boundary_precision_20260913.md`. This pause status is historical.

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
   inputs. Both observed collective outputs equal the FP64-summed local
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

## Second source: attention autotuning changes the head tile

With invariant wo_a enabled, layer1 on all eight ranks has identical Q,
positions and selected SWA KV bytes between cached decode and full recompute.
The captured single-scope autotune choices differ:

- Decode: BLOCK_H16 / BLOCK_N32 / 4 waves / 1 stage.
- Prefill: BLOCK_H64 / BLOCK_N32 / 4 waves / 1 stage.
- The recorded dual-scope choice on both sides is H16/N64/8 waves/1 stage.

The fixed-single-scope component gives matching M1/M204 attention and wo_a
outputs on all eight ranks. It reproduces cached service output on all eight;
the old prefill rank1 wo_a output differs in 21 elements. This is evidence of
a second M-dependent numerical path, not a finding of incorrect KV addresses.

Added `SGLANG_DSV41_ATTN_INVARIANT=0` by default. Only the HIP V4.1 model opts
in; radix dispatch requires the Triton backend and rejects unified KV. Fixed
mode bypasses autotuning, uses the recorded decode tile above for single
scope and H16/N64/8 waves for dual scope, and disables split-K. Default-false
dispatch retains the previous implementation. The V4 Flash model is untouched.
Three CPU mock tests check dispatch only, not GPU arithmetic. The startup CPU
suite passed 124 tests +16 subtests. The earlier 1000 graph-replay result is
for the wo_a component only: the new attention mode is not graph-certified.

## Combined new-process validation: improvement, not complete repair

Process ready 16:21:54 HKT, parent PID284493, both selectors enabled. All
other correctness-launcher settings remain as in Scope. For the same fixed
204-ID prefix, committed ID31151 agrees and max common Top20 logprob delta is
**0.37497711181640625**, versus control **1.2500200271606445**. This is about
70% smaller for this one probe; it is not an overall model-quality or speed
claim. Final normalized hidden relative L2 at this probe falls from 0.13688998
to 0.0442999. No matched performance trial was run.

Selected current-row module traces show:

- Layers0–12 main outputs agree; layer13 attention also agrees.
- Layer13 post-attention RMSNorm input already differs in one BF16 element,
  max 2.384185791015625e-7. Its output has the same tiny maximum difference.
  Do not call this an RMSNorm kernel fault: the input is different.
- Layer13 MoE output agrees, while Engram output differs in one element,
  max 1.1920928955078125e-7, despite matching captured input IDs/hidden.
- Layer21 attention still agrees. Its post-attention norm input differs by
  max 0.0001220703125; MoE output then differs in ten elements, max0.001953125.
- Layer22 attention is the first large persistent attention-output difference:
  4552 elements, max0.046875, relative L2 about0.0153085.

After resumption, the next bounded investigation is the HC collapse/post
boundary and Engram reductions around these first differences. Their exact
cause is **not established**. Do not start another experiment until the user
releases the GPUs after the emergency job.

## Final in-flight run completed before shutdown

France: `The capital of France is Paris.`; 8 completion IDs, natural EOS.
This is a semantic smoke check, not a sufficient numerical oracle.

The real event-sessionization SQL request generated **803 completion tokens**
and reached natural EOS. The extracted read-only query passed **16/16 SQLite
fixtures**, including empty input, equal timestamps and the strict >1800-second
session boundary. Only SQL results are functionally graded; prose is not a
formal oracle. The explanation is readable but has an imprecise sentence about
zero-based counting; the SQL explicitly adds one and passes the fixtures.

The same baseline continuation was teacher-forced at absolute prefix lengths
203/204/205, 255/256/257, 511/512/513, 767/768/769, twice per position:

- Coverage complete; **24/24 committed IDs and Top1 IDs agree**.
- Minimum Top20 overlap0.90; maximum common logprob delta **2.2050838470458984**.
- Thus agreement of the sampled next tokens does **not** establish bitwise
  logits, all-position parity or removal of long-context drift.
- Baseline completion SHA256:
  `dd3e1a4e9ffaca59be78da7f2e7e6847df055302dc0731278b58976b6a609242`.

New evidence in the existing experiment directory:
`single-scope-fixed-contract.json`, `single-scope-prefix-summary.json`,
`combined-row-comparison.json`, `combined-prefix-summary.json`,
`combined-france.json`, `combined-long-prefix-summary.json`,
`combined-sql-functional.json`, and `combined-long-prefix-http.tar.gz`.
The archive preserves raw request/response evidence, including baseline text
and Top20 logprobs; it contains no model weights. Large raw traces remain in
`/tmp/dsv41-single-scope-precision-20260913/` and
`/tmp/dsv41-combined-precision-20260913/`, not Git.

After the final request completed, sent SIGTERM only to verified service
parent284493. Confirmed the parent and eight scheduler children exited,
port30101 has no listener, and `amd-smi process --gpu all` reports no running
processes on GPUs0–7. User tmux `model` was preserved. No further GPU work or
service restart is authorized during this pause; no data was deleted.
