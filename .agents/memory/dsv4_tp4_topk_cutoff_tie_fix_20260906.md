# TP4 remaining drift: deterministic Top-512 cutoff ties

## Scope and evidence

Native TP4/EP1/no-A2A, original safetensors, chunk 2304, MFMA32/64
enabled, graph BS1, no scheduler overlap. Physical GCDs 4–7.
This follows the MFMA down wave-shuffle fix (719de619e1), logical
index order diagnostic (6c459b84b3), and preshuffled indexer cache
reader fix (bfb41502f6). None changes stored weight precision.

After the cache-layout fix, 15/16 diverse 2304-token code requests had
exact 32-token outputs and logprobs across three fixed-batch runs.
Request index 2 still varied, including when isolated at C1.

Captured all C4 prefill indexers for three fresh-cache C1 replays:
`/tmp/dsv4_tp4_request2_indexers/layer_{layer}_replay_{round}.pt`.
Comparing replay 0 with 2: layers 2/4/6 match, and layer 8 scores
match exactly, but logical Top-K membership differs on zero-based row
2144. That row has equal scores straddling the 512th selection cutoff.
The legacy AOT atomic final-bin selection is arrival-order-dependent.
Canonicalizing output order alone cannot fix membership nondeterminism.
Repeated isolated Top-K on this capture reproduces the membership change.

## Fix candidate

`topk_deterministic_hip.cuh`: one wave64-compatible 256-thread CTA per
row, radix threshold over a total 64-bit key (ordered FP32 score,
inverse logical ID). Equal scores prefer smaller logical positions.
Histogram atomics count only; they never choose the winning IDs.
Emission scans descending logical positions with wave ballots and
fixed block-prefix offsets. Both membership and final attention
accumulation order are deterministic. Physical page placement cannot
affect selection or order. No score perturbation, D2H, global task
queue, or bounded shared-memory candidate array.

Selector (default 2 in the TP4/EP1/no-A2A harness after validation):
`SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER=2`.
Mode 1 remains the earlier output-order-only diagnostic; mode 0 is
legacy AOT. Do not call mode 1 a complete determinism fix.

## Component validation

`scripts/rocm/check_dsv4_deterministic_topk.py` compares against an
independent CPU stable score sort, tie-breaking by logical ID, then
descending-ID emission. Tests random signed scores, all tied negative
scores, integer score ties, empty/short/ragged rows, strided scores and
page tables, 32K/64K tied rows, and the actual layer-8 M2304 capture.
All five cases pass 1000 graph replays each. Output buffers are
poisoned between replays, and physical slot transformation is checked.
Log: `/tmp/dsv4_deterministic_topk_oracle.log`.

## TP4 end-to-end gate

Three fresh-cache fixed-input C16 runs, each request 2304 input tokens
and 32 output tokens: **16/16 complete token sequences, token logprobs,
top-5 logprobs, and texts are exact across all three runs**. All cached
token counts are zero. This includes formerly drifting request index 2.
Artifacts: `/tmp/dsv4_tp4_topk2_c16_{1,2,3}.json`.
Compare with `scripts/rocm/compare_dsv4_output_replays.py`.

The isolated formerly drifting request index 2 also passes **3 x 256
tokens**: IDs, token logprobs and top-5 logprobs exact, cached tokens 0.
Artifacts: `/tmp/dsv4_tp4_topk2_c1_{1,2,3}.json`.

Extended C16 gate: **2 x 16 x 256 tokens**, every request's IDs, token
logprobs, top-5 logprobs and text match exactly, with no cache hits.
Artifacts: `/tmp/dsv4_tp4_topk2_c16long_{1,2}.json`.

France QA (official chat input IDs, greedy, normal EOS):
`The capital of France is **Paris**.`; 9 completion tokens, EOS ID 1.
The code request produces coherent code-analysis prose, but some model
interpretations (e.g. expanding MHC as Multi-Head Compression) are wrong.
Do not equate deterministic generation or a France sentinel with full
code-analysis correctness.

The harness restores MFMA32/64 defaults and enables mode 2 for
TP4/EP1/no-A2A only. TP8 and other EP layouts keep prior defaults pending
their own validation. Independent-process verification is pending.
Fixed-batch TP4 repeatability does not itself establish invariance to
arbitrary batching or equality to a different reduction algorithm.
