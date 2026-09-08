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
