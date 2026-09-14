# Bounded manual review — in progress

This records observations, not an answer-accuracy or global-determinism pass.
Final-control review remains pending. Both candidate waves have been read.

## A1 control

Read every128-token excerpt in `A1/quality-0.json`, then every changed text
in `A1/quality-1.json`, with responses aligned by request ID. The driver
verified32/32 full input echoes, zero cache hits,128 completion tokens and
tokenizer decoding against response text. Full-output repeat is6/16.

Changed cases and zero-based first divergence:
`0:28, 2:50, 5:18, 6:0, 8:2, 9:23, 10:12, 12:10, 13:3, 14:106`.
Thus15/16 first tokens match even though only6 full excerpts repeat.

The inspected control excerpts remain coherent and topic-related, without
obvious looping/garbling. Several assert alleged code defects without enough
evidence in the128-token excerpt to validate those assertions. Case8 again
includes the previously identified inaccurate dismissal of distributed helper
code as absent; the alternative wave analyzes the helpers. The shared actual
manifest contains their definitions, as documented in the preceding wide32K
trial. This factual-quality limitation is present before the20-iteration change.

Timing is separate: A1 three-wave median5542.583721 input tok/s, warmup
5407.299735. These are original-V4 TP8 C16x32K zero-prefix wave metrics,
with wide query reuse enabled,1M KV and legacy8-iteration batch1 MHC.

## B candidate, first quality wave (second wave and A2 still pending)

Read all16 excerpts from `B/quality-0.json`. Each is coherent, code-related,
and shows no obvious repetition collapse or garbling within128 tokens.
Case8 now discusses the distributed helpers rather than claiming they are
absent. This is a bounded observation, not proof that all alleged defects or
line references in these generated analyses are accurate.

An independent read-only JSON check verified all16 input echoes against the
manifest and all128-token completion counts. None of these16 candidate full
excerpts matches either A1 quality wave; first tokens match7/16 and6/16,
respectively. Do not describe this as byte-equivalent to legacy8 or claim
that normal-looking text proves numerical equivalence. The component oracle
already established a deliberate comb change from8 to20 iterations.

The B service log also records native decode raw_bs12 replaying a16-row graph.
This is direct evidence of dynamic execution shape, not proof that it caused
any particular output difference. Prefill policy alone does not control every
source of end-to-end non-determinism.

## B candidate, second quality wave

Read all16 `B/quality-1.json` excerpts, including all seven differing from B.0.
They remain coherent and source-topic-related without obvious collapse. The
same factual-accuracy limitation applies: these are truncated analyses, not
validated code-audit findings.

Both waves have32/32 identical input echoes, zero cache hits and128-token
completion counts. Full-answer repeat is9/16; first-token repeat is16/16.
Changed case / zero-based first difference:
`3:60, 4:46, 6:40, 10:51, 11:12, 13:24, 14:26`.
This does not establish statistically improved determinism over A1's6/16,
and it clearly does not eliminate drift. Final-control A2 is still required.

Candidate service completed and its owned process tree stopped. Formal leg
medians are5480.998651 and5480.220425 input tok/s; A2 is now loading.
