# C1-only TP8 wo_a screen, 2026-09-08

User requested C1 scope; the previous M2/M4 candidates are deferred.
TP8/EP1/no-A2A, original checkpoint, native AR, live parent2268456 on30011.
No production code/default changed after this screen; service was not restarted.

## Baseline

Three distinct fixed code tasks x3 measured repetitions,256 output tokens.
Warmup excluded; client HTTP wall time includes short prefill.
Overall median84.16982 tok/s; task medians84.16982/84.20808/83.96581.
France exact. All completions match prior B4 candidate; all6 full-prefix probes
match B4 input IDs/logprobs/output top-logprobs/next IDs exactly.
These are full-prefix prefill probes, not a cached-decode numerical oracle.
Artifact:/tmp/dsv4_tp8_c1_resume_A1_20260908.json; data preserved in adjacent JSON.

## Fixed-reduction geometry screen

Existing G1/M1/N1024/K4096 grouped kernel. Hold unroll2 fixed, change rows per
wave and waves per CTA. GPU4 only while service idle.100 mutations and10 graph
replays/mutation per configuration: all finite and exactly equal the current
HIP baseline. Three ABBA cycles per comparison,43-node graph bursts x10.
Two memory regimes: one repeated weight, and43 independent8MiB weight tensors.
The latter avoids timing only an L2-resident layer, but is not a full model.

| rows,waves | same weight us | 43 weights us |
|---|---:|---:|
|1,4 current|6.937|9.736|
|1,2|7.594|10.350|
|1,8|7.144|10.218|
|2,2|8.131|10.932|
|2,4|7.726|10.707|
|2,8|7.045|9.752|

Current geometry wins. No service experiment warranted for these candidates.

## Direct activation reads

An isolated HIP template option replaces shared activation staging/barrier with
vector global activation reads. Same dot8 accumulation, lane ownership, shuffle
tree and BF16 rounding.100 mutations x10 replays all exact/finite for all3
candidates. Best r1/u2/w4:6.9807→6.6399us hot;9.7280→9.1406us across43 weights.
Other direct geometries are slower. This is about0.0253ms per43-layer token if
fully critical, around0.21% of C1 wall time: not an E2E measurement or speedup.

Keep the candidate as `.agents/experiments/dsv4_tp8_c1_woa_direct_x_20260908.patch`.
The production header was restored exactly after the test; original unrelated
worktree edits remain untouched. To reproduce direct mode, apply that patch in
a controlled worktree before using `--direct-x` in the new screen script.
Plain geometry mode uses the unchanged production header.

CPU shape/dispatch guard regression:2 tests passed. No broader backend selector
was added. C1 remains the84.17tok/s validated baseline; no new E2E gain claimed.
Next optimization should seek a larger C1 stage budget, not expand this small
geometry sweep or enable M2/M4 under the user's C1-only instruction.
