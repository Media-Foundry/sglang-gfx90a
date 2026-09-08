# Remaining C32 decode slot effect

No production changes in this experiment. Service PID2987321 at commitd6a07c92da,
row-stable prefill candidate enabled, native TP8/EP1,1M KV pool, debug dumps off.

New `check_dsv4_same_prompt_decode.py` is a correctness diagnostic, explicitly
not a diverse throughput benchmark. It repeats an existing real code prompt,
uses independent cache salts, a client start barrier and completion-only IDs.
Workload source `/tmp/dsv4_tp8_c32_ar_code_workload_20260908.json`, index17:
"Given a CUDA kernel with uncoalesced column reads, rewrite its indexing for
coalesced memory access."

Three32-client256-token waves produce exactly two variants,16 copies each,
with identical variant multisets across all3 waves. Variants first diverge at
zero-based generated index8 (ninth token); per-client cross-round exact10/32
because clients are assigned different batch slots. This is evidence of a
deterministic position-dependent trajectory, not proof of a random race.
Raw `/tmp/dsv4_tp8_same_code_decode_20260908.json`.

Both256-token texts are coherent planning/explanations of coalesced accesses,
row-major column strides, thread mapping and transpose/shared-memory options.
No gibberish/loop was observed in these excerpts, but they are truncated
reasoning, not executable-code correctness or a complete answer acceptance.

Teacher-force the shared first8 continuation IDs back through prefill:
32 clients x3 waves all select next token304; the cached-decode variants had
selected304 or418 after exactly that same eight-token prefix.
Raw `/tmp/dsv4_tp8_same_code_forced8_20260908.json`.
This narrows the remaining difference to cached-decode/batch-position execution
versus recomputed prefill, not prompt wrapping. It does not identify a kernel.

C1 repeated twice produces identical full256-token IDs, but differs from C32
starting at zero-based generated index3. Thus its ninth token must NOT be
treated as a teacher-forced comparison to the C32 ninth token.
Raw `/tmp/dsv4_tp8_same_code_c1_20260908.json`.

Next: long diverse C32 generation and fixed-prefix cached-decode logits if
further isolation is needed. Do not promote the candidate on hash counts alone.

## Long diverse C32 screen

Same32 distinct code prompts as accepted short benchmark;2048 forced tokens,
two rounds. E2E920.3745/919.8167 tok/s, median920.0956; resident median925.2218.
All64 requests finish length with2048 tokens. Full cross-round exact10/32.
This is not directly comparable to the256-token~984 E2E result because the
context grows much longer. No baseline ABBA or speedup claim.
`/tmp/dsv4_tp8_rowstable_long_c32_20260908.json`.

The existing throughput harness only retained hashes/first16 tokens, so it
cannot support full semantic review. Added optional `--save-output-ids`
(off by default); it records IDs after timed execution, without changing the
request payload or generation. A separate long wave retains all completion
IDs for review. No generated code is executed during review.

The retained-output wave completed at920.2247 E2E /925.1357 resident tok/s:
`/tmp/dsv4_tp8_rowstable_long_c32_quality_20260908.json`.
All32 saved completion arrays have2048 IDs and reproduce their saved SHA256.
Review trims at the first EOS (14/32 reached EOS before the forced token limit);
post-EOS forced output is not counted as an answer quality signal.
No16-token n-gram repeats>=8 times before EOS in any of the32 samples.
This is only a degeneracy heuristic, not a general semantic pass.

Manual excerpts from requests0/7/17/23 were coherent discussions of transaction
rollback, persistent range-sum versions, coalesced CUDA access and a cancellable
Go worker pool.12 complete Python fences parse;2 fail because case0's `with`
bodies contain only comment placeholders, so those are not runnable examples.
The CUDA example changes a column-major indexing example to row-major and
therefore needs a layout/functional contract check; do not claim a semantics-
preserving rewrite based on prose alone. No model output code was executed.

Conclusion: the prior all-spaces/obvious-loop failure was not observed in this
screen. Full code correctness and equivalence against baseline are unproven.
Remaining cached-decode batch-position numerics are reproducible; no production
optimization is promoted here and no numeric difference is dismissed solely
because France passes.
