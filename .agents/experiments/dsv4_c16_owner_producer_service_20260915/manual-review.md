# Manual review scope

Read all16 B-check wave0 outputs and the two changed wave1 outputs (cases4,
15). Other14 wave1 outputs are byte-identical to reviewed wave0. Also inspected
control cases2/3/7/11/12/13 to understand substantive cross-config differences.

Candidate text is coherent and topical within256-token truncation, with no
observed repetitive collapse. This does not certify code correctness or
factual accuracy. Alternative claimed defects appear in cases2 and7. Case13
has questionable process-group creation reasoning in BOTH arms. Therefore
do not claim candidate is semantically equivalent based on coherence/France.

Control repeats16/16; candidate14/16. Cross-config whole-output equality1/16.
Proceed with fixed-prefix target numerical checks, keep default off. No E2E
performance claim from the instrumented candidate.
