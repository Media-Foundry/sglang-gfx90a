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
