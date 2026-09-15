# Bounded mixed-prefix review, in progress

Only completed output waves explicitly listed below have been read. This is
not a complete-answer factual-accuracy certificate or global determinism proof.

## A1 quality wave0

Read all16 full128-token excerpts from `A1/quality.json` round0. Input echoes
16/16 equal the manifest, each output has128 IDs, and actual cached-token
vector equals the planned0/25/50/75% prefixes. Text is generally coherent and
code-topic-related without obvious loops/garbling in the bounded excerpts.

**Retain an actual answer limitation:** case8 says the excerpt does not
contain the requested distributed-helper failure-handling code, while also
listing `broadcast_pyobj`, `point_to_point_pyobj`, `init_custom_process_group`.
Decoded the exact case8 input IDs with the local tokenizer: definitions begin
at character offsets80363,82748,90393 respectively, and their bodies are
present in the supplied excerpt. Thus a broad assertion that these helpers
are absent is not supported. This is control behavior, before wide-query is
enabled; it is not evidence the candidate broke semantics or a kernel is
the source of this reasoning limitation.

Other generated bug claims/approximate line references have not been audited
for factual accuracy. In particular, fluent assertions about batch filtering,
scale handling or backpressure are not established repository findings.

A1 quality wave1 and both candidate/final-control quality waves remain pending.

## A1 quality wave1 and completion (supersedes A1 pending note)

Read all15 changed excerpts in round1; case4 is token-identical to the already
read round0. Text remains coherent and on the supplied code topics without
obvious looping/garbling. Case8 now discusses the provided helper functions
instead of the prior absence assertion. This demonstrates control answer
variation; it does not establish which numerical operation caused it.

Complete128-token outputs repeat only1/16, first tokens12/16. First divergence
positions(case:zero-based index):
`0:21,1:0,2:41,3:10,5:0,6:9,7:50,8:0,9:12,10:10,11:0,12:55,13:3,14:9,15:61`.
Do not describe this as globally stable output. The multiple potential bug
claims in both rounds remain unverified, notwithstanding fluent wording.

A1 stopped with remaining owned PIDs=[]; B started separately. Candidate and
final-control quality review remain pending; no overall semantic verdict yet.

Independent local-tokenizer verification subsequently checked both A1 waves:
32/32 exact input echoes and32/32 output-ID decoding equal stored text,128
completion tokens each, identical planned/actual cache vectors. This check
does not certify the generated claims as true.
