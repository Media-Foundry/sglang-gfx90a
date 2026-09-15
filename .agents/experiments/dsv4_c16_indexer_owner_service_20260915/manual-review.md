# Bounded long-output review

Read all16 B/quality-0 texts and the only alternative text in B/quality-1.
All are coherent, on-topic source-code discussions; no repetitive/placeholder
collapse is visible within the128-token window. They are intentionally cut at
128 tokens, often mid-sentence/code. This is NOT verification that the generated
line numbers or claimed code bugs are factually correct, nor a long-horizon
quality benchmark.

31/32 candidate outputs exactly match at least one current A1/A2 control.
The remaining output is B.1 case8 (distributed-helper failure handling).
It changes a heading and explanatory phrasing, remains coherent, and exactly
matches the historical accepted-control A1/quality-1 response for the identical
prompt in `dsv4_c16_premix_pair_service_20260915/`.

Thus no new text branch appears in the combined current/historical evidence.
Current controls repeat16/16, candidate15/16: do NOT call the candidate output
globally bitwise deterministic or claim the new communication fixes drift.
The fresh-process audit already shows late-layer drift on identical inputs
without query ownership; that first-divergent operator remains to be found.

The diagnostic service additionally dual-computed full-local and owner
selection in the same forward with exact scores, logical IDs and physical IDs.
All16 diagnostic128-token answers matched the prior accepted B/quality-0 run.
Every service arm passed France and checked exact input echoes, zero KV hits
and completion-only token lengths/decoding.

`quality-review.json` retains its automatic `manual_review_completed=false`
field; this separate review closes the bounded manual gate without altering
the automatic analysis artifact. Raw output files and their hashes are retained.
