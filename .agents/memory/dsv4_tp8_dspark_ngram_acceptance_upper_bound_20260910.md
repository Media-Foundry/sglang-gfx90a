# TP8 DSpark output-copy n-gram acceptance bound (2026-09-10)

To test whether a cheap code-oriented copy proposer could raise the numerator
without increasing strict M128 target rows, the completed 32-request x 1024
token real-code outputs from both accepted draft-M96 checkpoint processes were
replayed offline.  At every simulated step, the proposer used only the already
committed output prefix, found the most recent previous occurrence of its last
N tokens, and copied up to three following tokens.  No future token was visible.

The production DSpark request metadata reported mean speculative acceptance
lengths 2.9384 and 2.9257.  Offline copy-proposer mean committed tokens/step:

| N | process B1 | process B2 |
|---:|---:|---:|
| 1 | 1.2643 | 1.2627 |
| 2 | 1.1557 | 1.1594 |
| 3 | 1.0951 | 1.1025 |
| 4 | 1.0630 | 1.0701 |
| 8 | 1.0186 | 1.0242 |

Even the optimistic N=1 copy rule is far below the learned DSpark draft.
Do not replace DSpark with output-only n-gram copying for this heterogeneous
code workload.  A hybrid would require the actual DSpark proposal/confidence
stream to establish complementarity; output repetition alone provides no
promising upper bound.

The same data clarifies the 2k feasibility constraint.  With mean acceptance
about 2.93 and current throughput about 1088 tok/s, gamma-three's absolute
full-acceptance bound at unchanged step time is only roughly
`1088 * 4 / 2.93 = 1485 tok/s`.  Reaching 2k therefore requires a material
target-step reduction plus a deeper/broader proposal structure; confidence or
acceptance tuning alone cannot satisfy the goal.

Source artifacts:

- `/tmp/dsv4_tp8_draft_m96_rowprefetch_abba_retry5_20260910/B1.json`
- `/tmp/dsv4_tp8_draft_m96_rowprefetch_abba_retry5_20260910/B2.json`

