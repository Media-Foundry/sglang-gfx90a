# Bounded output review — in progress

This is a manual inspection record, not the final ABBA acceptance. A2 was
still loading when the candidate review below was recorded. Final analysis
and the complete control comparison remain required.

## Candidate waves inspected

Read all 16 texts in `B/quality-0.json`, then the six changed texts in
`B/quality-1.json`. All unchanged second-wave texts were verified equal by
completion token IDs. Responses were ordered/checked by their request ID;
the source arrays also happened to be in case order.

- Both waves: 16/16 full `prompt_token_ids` match their manifest input IDs.
- Both waves: 128 completion tokens per response; no prefix cache hits.
- Same candidate across waves: 10/16 full completions exact, 16/16 first
  tokens exact. Changed cases and zero-based first divergence:
  `1:62, 3:22, 8:2, 9:23, 12:10, 13:23`.
- First candidate wave: 11/16 completions match at least one of the two A1
  control waves. Cases `1,5,8,13,15` match neither; those control texts were
  also read alongside the candidate texts.

The inspected excerpts are coherent code-review prose, on the requested
topics, without obvious repetition loops or garbling. This does NOT establish
the factual correctness of every alleged code defect, complete-answer quality,
or global numerical equivalence. In particular case 8 changes between an
analysis of distributed helpers and a claim that the excerpt lacks the relevant
handling code; the latter wording also occurs in A1. This is a substantive
answer variation, not merely punctuation, even though neither excerpt collapses.

A limited factual spot-check of case 8's actual manifest prompt confirms full
definitions of `broadcast_pyobj`, `point_to_point_pyobj`, and
`kill_process_tree` are present. Their bodies include the ROCm broadcast
barrier, receive/wait handling, and process-reap timeout handling respectively.
The excerpt does end partway through a later `if get_` line, but this does not
remove the preceding helper definitions. Thus the output variant dismissing
the relevant code as absent is not an accurate description of this input.
It also appears in A1, so it cannot be attributed to the query-reuse change
from this evidence. The bounded smoke label must not be promoted to a factual
answer-accuracy pass.

Hashes:

```
f41533679b85057e60cb5f9fe50fa62b20108e7e1c1776b58c80b36f9ef5d9d2  B/quality-0.json
ce12d5ed17043c8654bb5b50c90684485ee9aea9ec553957817d7813beebebf6  B/quality-1.json
```

## Separate one-token performance waves

A1 and B1 each vary at case 6 between token 39111 (`Looking`) and 372 (`##`),
while their other 15 cases have the same first token. Their input manifest
hashes agree. These one-token runs must not be confused with the two
128-token quality waves above, nor used alone to attribute drift to query reuse.

## Cold shape observation

Candidate warmup logged `reuse_runtime_m` compilation on all eight ranks,
8.42–8.73 seconds each. These are concurrent rank observations, not durations
to sum. Steady B1/B2 medians are 5545.8868/5545.7144 input tokens/s; the
warmup rate was 4971.0156. Wide-width prewarming is still an outstanding
production requirement. Final formal-window compile checks remain in
`analyze.py`.
