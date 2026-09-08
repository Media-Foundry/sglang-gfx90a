# TP8 M32 down uniform service integration

Component baseline db8c255564. Added default-off
`SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM`. The native M32 scope recognizes this
flag independently, without enabling legacy AR, gate-prefetch or attention
experiments. Runner eligibility requires native decode scope, gfx90a, TP8/EP1,
M32/H4096/Top6, E256/K256 packed down shape, A4/R2/W8/D832/LDS. Runtime-M,
row-prefetch, logical scales, split-MoE and deferred-finalize are excluded.
MFMA/consumer paths do not enter the standard down branch.

The public wrapper's keyword-only argument defaults False; its True path
asserts exact geometry/layout, compiles a distinct JIT module with the tested
macro, and uses the same run/partial/reducer API and allocations. No new cache,
weight conversion, communication, stream join or persistent tensor.

`check_dsv4_down_uniform_scope.py` executes the actual predicates/contextmanager
with CPU stand-ins: all tested tiers/native/decode exclusions, context cleanup,
independent enable, topology/shape exclusions, down-only call and default-off
checks pass. Python compile checks pass.

`check_dsv4_down_uniform_wrapper.py` tested the actual public Tensor-returning
wrapper on GPU4 against its default path:100 mutations of INT8 inputs, scales,
router weights, nonconstant physical weight scales and bijective expert-ID
relabeling. Final BF16 raw int16 bits match100/100 (including signed zero),
finite;1000 additional replays stable. This is not full model correctness.
AMD-SMI ownership validated before GPU work; baseline PID3391501 retained1M KV.

Service acceptance will use run_dsv4_rowstable_abba.py with candidate flag,
same fixed32 real code prompts, six256-token C32 waves/block, C1 full-token
reference checks and separate FranceC32 sentinel. A(candidate),B,B,A; 1M pool,
interleave-all, other accepted flags fixed. Summarizer accepts the new flag.
No performance default or checkpoint promotion before completed ABBA.

Planned state: /tmp/dsv4_tp8_down_uniform_abba_20260908.json.
Controller leaves final candidate process resident; if rejected, remove only
this flag and restore the validated baseline. Do not reuse a prior PID blindly.

## First service ABBA completed

Implementation ae0812c268, controller session75412 exited0. Candidate processes
3422116/3441083, control3433069 reused for B/B. All four blocks completed.
First capture included header-triggered compilation:89.61s TP0; subsequent
control/candidate captures11.97/12.11s. All TP0 graph0.65GB, available15.64GB,
runtime max_total_num_tokens1048576. Candidate logs344 hits each; control0.
The working tree retains unrelated user changes throughout; not a clean-checkout
performance claim. No relevant model edits during the four blocks.

Warm geomean across the two blocks of each arm:

| Metric | Control | Candidate | Change |
|---|---:|---:|---:|
|C1 HTTP tok/s|84.72708|83.50741|-1.4395%|
|C32 HTTP tok/s|983.37402|989.20510|+0.5930%|
|C32 resident tok/s|1029.33469|1035.89893|+0.6377%|

Discard C32 wave0, use five warm waves/block. C1 has two measured repetitions
per each of three cases, no trimming at this count. C1 is excluded by selector
but its observed slowdown must not be ignored or attributed without evidence.
This prevents default promotion on the first ABBA despite C32 improvement.

Correctness:24 measured C1 completions equal reference; six fixed-prefix
probes/block have matching IDs/input logprobs/top logprobs across the run.
These probes are prefill recomputations, NOT cached teacher-forced decode.
FranceC32 first9 answer-through-EOS IDs32/32 in every block. All768 code
completions have expected length/finish and recomputed hashes. Cross-round
exact requests5/6/12/9 out of32; neither arm proves full C32 determinism.
Adjacent ABBA JSON preserves summary, every raw C32 rate, C1 samples and
artifact SHA256; full IDs remain in the referenced local result files.

Repeat started with c1-rounds4 to increase C1 sensitivity, same six C32 waves,
same fixed manifest and pool. State:
`/tmp/dsv4_tp8_down_uniform_repeat_abba_20260908.json`, exec session61092.
First A reuses final first-run candidate PID3441083; later B and A are new
processes. Thus repeat is not wholly independent of the first experiment.
Follow the live controller; no automatic restart on an observation timeout.
Default remains False. Restore utility now accepts this exact flag as well
as deferred-finalize, still requiring completed state, matching live PID,
exact TP8/EP1/1M/loopback/model command and exclusive GPU ownership.

## Repeat completed: do not promote the joint C1/C32 default

Controller61092 exited0. Candidate A0 reused3441083; new control3449939
served B/B; new final candidate3459680. Four measured C1 reps/case/block,
per-case min/max trimmed, then block/arm geomeans. Same six C32 waves with
wave0 discarded.

| Metric | Control | Candidate | Change |
|---|---:|---:|---:|
|C1 HTTP tok/s|84.24832|83.38133|-1.0291%|
|C32 HTTP tok/s|984.55658|989.47957|+0.5000%|
|C32 resident tok/s|1030.50820|1036.19321|+0.5517%|

The small C32 gain repeats, but the observed C1 loss also repeats. This fails
joint default acceptance. Retain the opt-in experiment for diagnosis; do not
claim a general performance checkpoint or dismiss C1 because its predicate
does not select the new kernel. No proven mechanism for C1 slowdown yet.

Correctness:48/48 measured C1 outputs equal reference; fixed-prefix recompute
probes6/6 per field/block match (not cached decode oracle). All four FranceC32
answer-through-EOS prefixes32/32. All768 real-code responses pass length/finish/
hash validation; cross-round exact5/6/8/6 of32, not full determinism. C32
selection log344 in candidate,0 in control. New TP0 captures12.04s control,
12.06s candidate, both0.65GB/15.64GB available, actual pool1048576.

Read-only /proc module audit after benchmarks started: control TP0 PID3450230
and candidate TP0 PID3459945 each map58 SGLang gfx90a JIT modules.57 share
identical paths and binary SHA256. The sole replacement is the expected M32
down module. This excludes a different SGLang JIT binary for C1 in these
inventories; it does not inventory every AIter/BLAS binary, establish identical
allocation addresses, or explain the latency difference. Full inventories and
raw samples/artifact hashes are retained in the repeat JSON.

Restoration started with exact final PID3459680 and completed repeat state,
removing only DOWN_UNIFORM. Restore state:
`/tmp/dsv4_tp8_down_uniform_restore_20260908.json`, controller9543.
Wait for validation before calling the service restored. No other optimization
flag is removed, and the1M pool remains in the cloned command.
