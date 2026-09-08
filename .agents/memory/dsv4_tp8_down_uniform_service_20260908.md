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
