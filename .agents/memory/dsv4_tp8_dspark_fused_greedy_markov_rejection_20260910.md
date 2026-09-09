# TP8 strict DSpark fused-greedy Markov screen (2026-09-10)

## Scope

- Original DeepSeek-V4-Flash checkpoint weights.
- TP8 / EP1 / no A2A, strict full-target gamma three; target remains M128.
- Thirty-two heterogeneous code requests from
  `/tmp/dsv4_open_code_pd_20260908/decode.json`, 1024 output-token cap.
- The accepted target M128 and draft-only M96 row-prefetch paths remained on.
- Both arms used greedy-only draft capture (`SGLANG_DSPARK_FOLDED_SAMPLING=0`).
  The sole B-arm change was `SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV=1`.

This is distinct from the older folded-sampling-off rejection: that experiment
left the fused Vanilla Markov selector disabled, so it continued through the
ordinary three-step block sampler.  The B service here logged that the greedy
proposal was folded into the draft CUDA graph, proving the candidate was hit.

## Correctness and performance

Both arms answered the France sentinel with `Paris`.  The benchmark emitted
eight resident-window samples per arm (two groups of four):

```text
A, ordinary greedy Markov:
1090.67 1098.79 1097.87 1087.26 1083.49 1074.02 1091.58
reported center: 1089.7041 tok/s

B, fused greedy Markov:
1081.96 1101.92 1094.54 1078.44 1102.89 1090.39 1097.71
reported center: 1092.2878 tok/s
```

The observed difference is only +0.237%, much smaller than ordinary service
variation and far below the checkpoint threshold.  A full A/B/B/A service
sequence was stopped after this screen because the candidate cannot materially
advance the 2k objective even if the small center movement were real.

The greedy-only profile also reduced the admitted full-token pool to 743,424
despite requesting 1,048,576.  It therefore does not preserve the accepted
one-million-token capacity contract.

## Decision

Reject this configuration and keep the production profile unchanged.  Do not
enable fused-greedy Markov merely because the code path is reachable: its
proposal-tail saving is hidden by the complete draft/target graph schedule,
and the required folded-sampling-off configuration loses KV capacity.

Artifacts:

```text
/tmp/fused_markov_a1.json
/tmp/fused_markov_b1.json
/tmp/fused_markov_a_france.json
/tmp/fused_markov_b_france.json
```
