# C16 wo_b row-placement drift: causal component and guarded repair

Base `b3603ec801`, original DeepSeek-V4-Flash, TP8/EP1, native AR,
original weights and 1M logical KV. This is correctness work, not a speed win.

## Frozen service reproduction

`wob_oracle.py` reads layer0 snapshots from the completed
`layer0-stable-qkv-wqb` run. It reconstructs the actual local wo_b BF16 weight
from checkpoint FP8/E8M0 block128 tensors: full [4096,8192], column-sharded
into eight [4096,1024] matrices. Runtime BF16 caching and F.linear were checked
in `deepseek_v4.py` / `fp8.py`. Only physical GCD4 executes these eight fixtures.
The full M32768/M32767 tensors restore retained rows at their original offsets;
other rows are zero. Reproduction is required before interpreting drift.

All retained service outputs match F.linear exactly on both arms/all eight
shards. The corresponding 153 case15 input rows are bitwise identical before
wo_b. Across shards, the placement change reproduces **40 changed elements
in 32 rank/row pairs**. Padding the M32767 input back to M32768 preserves these
40 differences: changing M is unnecessary. Conversely, keeping each source at
the same row while changing only M produces **zero** sampled differences.
This isolates row-placement-sensitive GEMM rounding from upstream attention,
collectives and scheduler execution for this retained fixture.

The fixed M128/N128/K128, eight-wave implementation has zero shifted elements
and passes 100 varied-row perturbations per shard (800 total). A second screen
calls the actual integrated project entry and checks its full output against
the offline tile before repeating the tests. It also passes all 800 mutations.
These comparisons establish row invariance, not parity with the old GEMM:
fixed-tile accumulation changes other BF16 output values as expected.

Component median across shards/shapes: library **2.2444 ms**, fixed tile
**2.7309 ms** (about 22% slower). No speed gain is claimed. Source tensor hashes
are recorded in the component JSON; full weight/activation tensors stay local.

`wob_tie_analysis.py` recomputes the 40 changed dots in CPU FP64. Unlike the
earlier QKV fixture, **not all** are near a BF16 midpoint:36 pairs are adjacent
BF16 values, four are not; three references are exact midpoints and19 are
within1% of the observed pair spacing. The maximum midpoint distance is
2.0862e-7. One near-zero dot differs by multiple BF16 steps and neither arm
matches the rounded FP64 reference. Describe this as row-sensitive FP32
accumulation/rounding, including cancellation-sensitive outputs, not simply
"40 midpoint ties" or a claim that either library arm is the exact reference.

## Narrow service change

`SGLANG_DSV4_DEBUG_PREFILL_WOB_STABLE` defaults OFF and uses the existing
native/gfx90a/TP8/EP1/M8192..36864 diagnostic predicate. It requires an unbiased
cached-BF16 projection and reuses the already tested K1024/N4096 wq_b tile.
Existing collective selection is retained. Deferred and hidden-shard output
branches remain unchanged; decode/draft/small-prefill do not opt in.
Logging distinguishes wq_b and wo_b. Eight CPU contract tests passed.

The new flag is explicitly zeroed in the performance harness, so future C16
64K performance runs cannot silently inherit this slower numerical diagnostic.

## Completed service validation

`run_changed_rows.py --stable-qkv --stable-wqb --stable-wob` runs the same
warmup/A1/B1/A2 input-echo and layer0 evidence protocol, adding only wo_b to the
previous QKV/wq_b/wo_a + FP32-collective diagnostic profile. The process
completed all four waves, all64 input echoes and completion-ID/text checks,
then stopped its owned service (PID1028183). All eight ranks logged stable wo_b
selection. The retained final prefill group is M32768 cases12/13/14/15 versus
M32767 cases12/13/2/15; earlier groups are not covered by these snapshots.

All retained common-query attention stages, including wo_b partial, wo_b AR,
attention output, FFN-entry MHC and normalized FFN input now agree exactly.
Full retained QKV/KV comparisons also agree on all ranks. Same-order sampled
stages are exact on all ranks. `validate_wob.py` passes these bounded checks.

Remaining layer0 divergence is inside FFN: one sampled router-logit row per
shown rank differs (rank0/1 max3.5763e-7), without changing sampled TopK IDs or
weights. Shared expert output changes on several rows even with identical FFN
input; routed output remains exact on the samples. The replicated FFN output
has **eight** changed sampled rows, down from10 before the wo_b repair.
Do not claim those remaining shared/router causes proven before their own
frozen-input oracle. Complete128-token output parity is15/16 for same-order
A1/A2 and14/16 for reordered A1/B1. This is not whole-model determinism, and
the sampled layer0 exact repeat does not explain the later final-token split.

Evidence: `wob-evidence.tar.gz`,36 files,4,180,969 bytes, SHA256
`ad259156206240b30a0a1e696aee14d782718cd6ba4f16e16e896180c44bf759`.
Includes component/tie results, service logs and input/output evidence, source
patch, and hashed activation manifest. Full activation/weight tensors are not
uploaded. All numerical selectors remain default-off; C16 throughput config
and the +1.82% result are not changed by this diagnostic.

## Fork-main handoff

The user explicitly requested working on **our fork's main**, not upstream
sgl-project. Remote gfx90a/main was4abec06ecc and was an ancestor of the tested
working history (92 commits behind before this repair). It can fast-forward.
Preserve that tip in `backup/fork-main-before-sync-20260914`; preserve the old
local bring-up main505b337379 in `backup/local-main-before-fork-sync-20260914`.
The working branch is now named main and tracks gfx90a/main. No force update
or upstream-origin write is needed. The generated graph-memory pickle remains
unstaged and unrelated.
