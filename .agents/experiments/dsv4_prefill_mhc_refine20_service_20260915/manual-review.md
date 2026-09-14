# Completed bounded review: full20 versus exact8+12

This is NOT the preceding legacy8-versus-config20 policy trial. All arms
here use config20 in ordinary prefill. The chronological notes below retain
their then-pending status; all six quality waves are now reviewed.

## A1, first quality wave

Read all16 `A1/quality-0.json`128-token excerpts. They are coherent,
code-topic-related and show no obvious loop/garbling in this bounded sample.
The assertions about alleged code defects and approximate line numbers are
not factual findings from this review. The distributed-helper case discusses
the functions, without the old claim that they are absent.

Independent JSON check: all16 input echoes equal the manifest, all counts
are128 completion tokens, and all cache-hit counts are zero. Final analysis
must still check tokenizer decoding, repeated outputs and the remaining arms.

A1 formal three-wave median5485.070096 input tok/s; warmup5352.748100.
Do not treat this partial control measurement as a refinement speed result.

## A1 second wave and process completion

Read the six changed excerpts in `A1/quality-1.json`; the other ten are
token-identical to the first wave already reviewed. All remain coherent and
code-topic-related without obvious collapse. Changed case / zero-based first
difference: `5:44,6:40,7:38,10:51,13:3,14:26`.

Input echoes32/32 match; full-answer repeat10/16, first-token repeat16/16.
This contemporaneous full20 control is still not globally deterministic.
A1 stopped with no remaining owned processes; candidate B has started with
both config20 and comb-refine enabled. Candidate review remains pending.

## B1 timing-wave input/token check (not long-output review)

The completed A1 and B1 timing manifests are equal. Each has three one-token
waves with16 zero-cache-hit requests. All A1 internal wave pairs match16/16
first tokens; all nine A1/B1 wave pairs also match16/16 first tokens.
This is encouraging for the exact implementation change, but cannot replace
the pending128-token candidate review or establish global determinism.
B1 median5537.794461 versus A1 5485.070096 input tok/s; B2/A2 still pending.

## B2 timing and first candidate quality wave

B2's three-wave median is5536.179571 input tok/s. All nine A1/B2 timing-wave
first-token comparisons also match16/16. The final A2 control remains pending.

Read all16 `B/quality-0.json`128-token excerpts. They are coherent, related to
the supplied code tasks, and show no obvious repetition collapse or garbling.
This does not validate the generated claims of bugs or their line references.
Independent input check:16/16 full echoes match, zero cache hits and128
completion-token counts. First tokens match16/16 against both A1 quality
waves. Full excerpts match10/16 and9/16 respectively; cases2,4,6,11 are not
identical to either A1 wave and have been read as part of the bounded review.
Candidate second-wave and final-control review are still pending.

## B second wave and process completion

Read all eight changed B.1 excerpts; the other eight are token-identical to
the already reviewed B.0 text. They remain coherent and code-topic-related,
without obvious looping/garbling. Changed case / zero-based first difference:
`2:41,4:46,5:44,6:13,10:51,11:12,13:3,14:26`.

Input echoes32/32 match with zero cache hits and128 completion-token counts.
First tokens repeat16/16, complete excerpts8/16. This is not a global
determinism fix; comparison with A1's10/16 alone is not enough to attribute
any drift to the candidate. Final A2 control remains required.

B finished and its owned process tree stopped. A2 is now loading with
config20 enabled and comb-refine disabled. No candidate throughput promotion
has been made while this last control remains outstanding.

## A2 and final bounded verdict

Read all16 A2.0 excerpts, plus the six changed A2.1 excerpts (cases
6,7,10,11,13,14); the other ten are token-identical to already reviewed text.
All are coherent and code-topic-related, without obvious looping/garbling.
Claims of code defects and line numbers remain unverified; these truncated
128-token excerpts are not a complete-answer accuracy evaluation.

The final analyzer validates96/96 full input echoes, zero cache hits,
completion counts, tokenizer decoding, all eight rank selectors, matching
production source and driver hashes, and stopped owned processes. Within-arm
full-text repeats are A1=10/16, B=8/16, A2=10/16; all have16/16 first-token
repeat. Thus global drift persists, and this trial cannot attribute its cause
to the exact local refinement implementation.

Full ABBA center:5482.727661 ->5536.987016 input tok/s (+0.989642%),
request TTFT -0.972128%. Formal legs contain no recorded slow-compilation
warnings. This is a gain relative to full20, not relative to legacy8, and
does not raise the accepted8K checkpoint. Keep the experiment opt-in pending
a separate decision on the underlying config20 policy.
