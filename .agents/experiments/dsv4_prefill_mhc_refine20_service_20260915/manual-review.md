# Bounded review in progress: full20 versus exact8+12

This is NOT the preceding legacy8-versus-config20 policy trial. All arms
here use config20 in ordinary prefill. Candidate and A2 remain unreviewed.

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
