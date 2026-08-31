# DeepSeek-V4-Flash TP4 DSpark gamma=5 / compact M128 graph experiment (2026-09-01)

## Scope and invariants

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- Target: TP4 / EP1 / no A2A, original checkpoint weights
- Workload: 32 distinct token-ID code/chat prompts from
  `.agents/memory/dsv4_tp8_diverse_32_input_ids.json`, 256 generated tokens,
  `stream_interval=1`, greedy requests
- Correctness gate: France first-nine exact + semantic Paris; all 32 requests
  must return 256 tokens with `finish=length`
- Candidate is strictly DSpark-only. Native AR cannot observe any of the new
  switches or the M128 anchor mask.

## Why this was attempted

Gamma=5 with a fixed 96-draft-token budget gives 32 anchors + 96 selected
draft rows, so the target can retain the same M128 shape as the accepted
gamma=3 profile while offering a longer proposal. The eager oracle had shown
mean committed length above 3.5, which justified implementing exact compact
ragged graph tiers rather than capturing the wasteful rectangular M192 tier.

## Infrastructure added

- Exact/extra compact-ragged target token tiers:
  `SGLANG_DSPARK_RAGGED_VERIFY_TOKEN_BUCKETS` and
  `SGLANG_DSPARK_RAGGED_VERIFY_EXTRA_TOKEN_BUCKETS`.
- Ragged decode graph capture iterates token tiers rather than interpreting
  them as request batch sizes; runner buffers are sized by the largest compact
  token tier.
- Optional `SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH` keeps the target graph but
  executes only the DSpark draft model eagerly.
- The M128 anchor-only routed mask uses live `qo_indptr_device` starts, so it
  works for non-uniform per-request verify lengths.

All switches default off.

## Two graph-capture bugs found

1. `mask[rows] = True` materialized a pageable host scalar during CUDA graph
   capture and failed with a CPU-to-CUDA copy error.
2. Replacing it with PyTorch `scatter_` captured successfully, but HIP graph
   replay raised an HSA hardware exception in the generic scatter/gather
   kernel. A conflict-free device equality reduction over 128 rows x 32 starts
   is stable and bounds-safe.

After the fix, target M128 capture consumed only 0.58--0.59 GiB/GCD and left
4.63--4.71 GiB/GCD free. Draft M160 capture consumed another 0.27--0.43
GiB/GCD. Earlier suspected OOM was therefore a misdiagnosis.

## End-to-end results

All four candidates passed the France gate and completed all 32 heterogeneous
requests at exact length.

| Candidate | Resident BS32 tok/s | Mean accepted length | Result |
|---|---:|---:|---|
| gamma=3 accepted control (recent median) | ~1045.5 | ~2.3--2.6 | control |
| gamma=5 M128 target graph + eager draft | 959.09 | 3.294 | reject (-8.3% vs control) |
| gamma=5 full graph, folded sampling | 706.09 | 2.316 | reject |
| gamma=5 full graph, greedy TP-local | 799.13 | 2.530 | reject |
| gamma=5 full graph, standard greedy | 794.92 | 2.629 | reject |

Evidence files from the run:

- `/tmp/dsv4_gamma5_m128_graph_r1_v2.json`
- `/tmp/dsv4_gamma5_m128_fullgraph_r1.json`
- `/tmp/dsv4_gamma5_m128_graph_greedy_r1.json`
- `/tmp/dsv4_gamma5_m128_graph_standard_greedy_r1.json`

## Conclusion

The compact M128 target graph is now functional and memory-efficient, but the
gamma=5 strategy is not a performance checkpoint. Even its best variant loses
to gamma=3, and draft graph replay reduces proposal acceptance relative to the
eager draft under the same target/budget. Disabling folded sampling and the
TP-local greedy shortcut does not restore eager acceptance, so the remaining
issue is a broader eager-vs-graph draft input/metadata parity bug.

Do not enable gamma=5 or these switches in the production TP4 BS32 profile.
If revisited, build a fixed-input draft-proposal oracle comparing eager and
graph `raw_hidden`, base logits, and per-step Markov tokens before another E2E
service sweep.

## Gamma=4 follow-up

A gamma=4 compact run used the same M128 target tier with a 0.75 fixed budget
(32 anchors + 96 draft rows) and eager draft. It also passed France and all
32 heterogeneous 256-token requests, but achieved only 929.11 resident BS32
tok/s with mean accepted length 2.849. This is 11.1% below the gamma=3 control
and below the gamma=5 eager-draft result. The temporary width-five M128 anchor
guard was therefore reverted; gamma=3 remains the accepted operating point.

Evidence: `/tmp/dsv4_gamma4_m128_eager_draft_r1.json`.

## Folded-proposal isolation

One additional gamma=5 run kept both target and draft transformer graphs but
set `SGLANG_DSPARK_FOLDED_PROPOSAL=0`. Base logits and Markov proposal were
therefore recomputed eagerly from the graph-produced draft hidden states. It
passed France and all 32 request-length gates, but reached only 782.09 resident
BS32 tok/s with mean accepted length 2.553.

This rules out the folded proposal tail hook as the primary acceptance loss.
The M160 draft transformer graph replay itself (its inputs, attention metadata,
or a graph-unsafe model kernel) differs from eager execution. The next useful
artifact is a fixed-input first-divergence oracle over draft `raw_hidden`, not
another E2E proposal switch.

Evidence: `/tmp/dsv4_gamma5_m128_graph_unfolded_r1.json`.
