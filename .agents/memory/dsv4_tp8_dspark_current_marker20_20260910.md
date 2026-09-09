# TP8 strict DSpark current layer-20 marker (2026-09-10)

## Scope

- Commit `745de4f232` plus the existing unrelated dirty-worktree changes.
- Original DeepSeek-V4-Flash weights, TP8/EP1/no-A2A.
- Strict full-target DSpark gamma three, target M128, draft M96 row-prefetch.
- Production folded sampling and accepted target paths were retained.
- 32 heterogeneous code requests, C32x256, one excluded warm group and one
  diagnostic group.
- Only layer 20 used graph-only, nonblocking gfx90a realtime markers every
  eight replays. Marker throughput is diagnostic, not an accepted speed result.

## Updated critical-path budget

The diagnostic group had 66 replay IDs matched across all eight ranks and 528
valid rank samples. Rank-max medians were:

| span | current us | prior 2026-09-09 us | change |
|---|---:|---:|---:|
| whole layer | 1568.96 | 1956.48 | -19.8% |
| attention MHC/norm | 116.88 | 117.84 | -0.8% |
| attention prepare | 209.84 | 210.24 | -0.2% |
| attention core | 166.32 | 236.56 | -29.7% |
| attention output/collective | 153.28 | 154.64 | -0.9% |
| FFN MHC/norm | 116.48 | 118.56 | -1.8% |
| MoE/collective | 828.48 | 1138.88 | -27.3% |

Detailed MoE rank-max medians from the printed marker slots were approximately:

```text
entry                              1.44 us
router                            42.08 us
Top-K                             15.84 us
routed expert main span          703.44 us
small post spans             4.16/5.28 us
final collective                  66.40 us
```

The detailed labels follow the same slot mapping used by
`dsv4_tp8_dspark_2k_marker20_20260909.json`; independently selected medians
must not be summed as an exact critical path.

The marker-perturbed diagnostic benchmark reported a center of 1108.36 tok/s
with individual resident samples 1095--1120 tok/s. This is not a production
checkpoint because timestamp kernels perturb scheduling.

## Implications

The accepted attention and routed-MoE changes are active and removed roughly
388 us from this representative layer relative to the older marker. Routed
experts remain the largest single span at about 45% of whole-layer time;
attention prepare is the second largest isolated span.

Two tempting routed scheduling ideas must remain closed:

- A4/A8 occupancy hybrid was exact but made the full routed stage 12.2% slower.
- expert-row persistent A4 was exact but made the full stage 27.8--29.6% slower.

The next routed experiment must change the gate-to-down dataflow (for example,
producer/consumer pipelining or a fused quantization boundary), not merely
reorder the same A4 tasks. The C4 Q2xH8 tail-sharing analysis has only about a
12.5% safe tile-reduction upper bound across the complete target batch and is
therefore a secondary, stackable attention candidate rather than the 2k main
line.

Artifacts:

```text
/tmp/dsv4_tp8_current_marker20_20260910.log
/tmp/dsv4_tp8_current_marker_diag.log
/tmp/dsv4_tp8_current_marker_summary.json
/tmp/dsv4_tp8_current_marker_detail.json
/tmp/dsv4_tp8_current_marker_warm.json
/tmp/dsv4_tp8_current_marker_diag.json
```
