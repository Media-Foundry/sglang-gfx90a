# Bounded manual review — in progress

This records observations, not an answer-accuracy or global-determinism pass.
Candidate and final-control review remain pending.

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
