# TP8 BS1 one-group wo_a GEMV (2026-09-07, in progress)

## Baseline and finding

Base commit `f4ecf87c6f`; native AR TP8/EP1/no-A2A with original checkpoint.
M36864 BF16-CK prefill profile, pool131072, memory0.80 and graph tiers
1/2/4/8/16/24/32 remain unchanged. No profiler/markers in performance arms.

The model config has `o_groups=8`, `o_lora_rank=1024`. TP8 therefore supplies
`x=[1,1,4096]`, `weight=[1,1024,4096]` to wo_a. The existing wave64 wrapper only
accepted two groups (TP4); despite its enabled environment variable it returned
None for TP8, falling back to `torch.einsum`.

The existing HIP kernel is already templated on group count. Parameterize the
JIT entry by groups (default2 preserves existing users), allow one group only
when explicitly authorized, and gate the model call on the opt-in environment
`SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV=1`, attnTP8, BS1, and native DECODE.
The wrapper additionally requires M1/G1/K4096/N1024 and the existing BF16,
contiguous, gfx90a checks. Prefill/verify/draft and other tiers do not select it.
The environment default is **False** pending acceptance.

## Component screen

Only physical GPU7, while the resident service was idle. No clock/OS changes.
`scripts/rocm/bench_dsv4_tp8_bs1_woa.py` uses static graph bursts with mirrored
arm order and mutable inputs/weights. Result:
`/tmp/dsv4_tp8_bs1_woa_micro_20260907.json`.

| Implementation | Graph median us |
| --- | ---: |
| Existing einsum fallback | 30.909 |
| Existing linear wave64 GEMV | 6.689 |
| Existing grouped kernel, G=1 | 6.833 |

Both wave64 variants: 100/100 stable mutation/replay checks (10 replays per
mutation), 70/100 entire output tensors bit-identical to einsum, max absolute
BF16 output difference 0.5 on random unscaled weights, max relative L2 vs FP32
matmul approximately 0.001788. This is not bitwise-equivalent arithmetic.

CPU unit tests cover the actual model predicate over decode/extend/verify/draft/
idle and TP4/TP8/BS1/2/32; wrapper tests cover preserved TP4 M1/M8 and rejected
M2/G1, M9 and unsupported groups.

## B1 service evidence

Fresh B1: `/tmp/dsv4_tp8_bs1_woa_B1_20260907.{log,json}`.
France exact. Three real code tasks, one warmup each, three measured rounds.
Most rates approximately83.5 tok/s; one62.508 tok/s sample is retained.
All completed repetitions share each task's candidate completion hash.

Relative to old baseline, reverse-linked-list first divergence is generated
token129; duplicate-email SQL first divergence token152; merge-arrays response
is unchanged across all256 tokens. The inspected outputs are coherent. The
generated reverse function (isolated AST with no calls/imports) passed four
linked-list cases; first SQL query passed an in-memory duplicate-email fixture.
This is limited semantic validation, not comprehensive code-quality validation.

**B1 did not complete all teacher probes:** after two probes it stalled on a
later prefill transition; GPU0 stayed100% active with idle peers. Similar stalls
preceded this candidate in the diagnostic screens. Cause is not established;
do not silently count this arm as full correctness passed. Partial JSON is
preserved with status=running. The stalled service tree was explicitly checked
and stopped, then all GPU processes were confirmed cleared.

## Independent B2 and long-source limitation

`/tmp/dsv4_tp8_bs1_woa_B2_20260907.json` completed: approximately83.1–83.3
tok/s, France exact, all nine measured256-token output ID sequences equal B1.
All six fixed-continuation prefill probes have identical input logprobs, output
top-logprobs and next output IDs to the baseline. Those are prefill probes:
the new selector does not run there, so they do **not** prove cached-decode
logits bitwise equal the old GEMM path.

`check_dsv4_long_source_response.py` uses a real2304-token source prompt, a
1024-token limit and normal EOS stopping, not forced post-EOS continuation.
The first B2 response completed with460 tokens in6.788s. Its prose was coherent
but included unsupported claims about bitsandbytes double quantization; do not
call this a verified code-audit answer. The second same-input/fresh-cache request
stalled. GPUs1/2/4/5/6/7 were100% active while0/3 were idle at the final check.
No root cause is established. The partial artifact is
`/tmp/dsv4_tp8_bs1_woa_B2_long_20260907.json`.

Thus the short-request decode speedup is promising but the candidate remains
**default-off**. A closing baseline, including the same long-source test, is
being restored to distinguish an existing serving/collective transition problem
from candidate-specific behavior. No default promotion or full stability claim.

## Separate negative TopK screen

The installed AIter lacks `topk_gating`; current selection uses the SGLang
Triton fallback. `bench_dsv4_bs1_topk_native.py` compared that fallback with the
existing native sqrt-softplus TopK on one GPU. Triton3.140us vs HIP9.954us;
100/100 IDs matched, but weights were bit-exact in only5/100 mutations.
At all logits=-40, max weight discrepancy was0.25 (not ordinary roundoff).
The native kernel uses log1p(exp(x)), which needs a detailed reference-contract
audit for such tails before reuse. No selector or production TopK was changed.
Artifact: `/tmp/dsv4_bs1_topk_native_20260907.json`.

## Closing A2 and isolated shared-read fallback A3

The closing A2 service (GEMV disabled, old fallback) completed all nine short
requests at approximately77 tok/s except two58–59 tok/s samples, then stalled
after three teacher probes. Consequently the transition stall is not exclusive
to the GEMV candidate. Its artifact remains partial:
`/tmp/dsv4_tp8_bs1_woa_A2_20260907.json`.

The runner declared IN_REPLAY shared reads but fell back to PRE_REPLAY when
an external event was unavailable. HIP does not create that event. Change the
fallback to POST_REPLAY to honor the declared read lifetime conservatively.
This is a synchronization contract fix, not yet proof of the stall root cause.
The existing shared-read fence tests pass (4 cases), and the isolated fallback
contract test passes.

A3 runs this fix alone, with the GEMV candidate OFF. The complete artifact
`/tmp/dsv4_tp8_war_post_A3_20260907.json` reports per-task medians76.818,
76.571 and76.820 tok/s. All nine measured completion ID sequences match the
old baseline; all six full-prefix teacher probes match on input IDs, input
logprobs, output top-logprobs and next IDs. France passes.

The following two fresh-cache2304-token source requests also both completed,
with482 completion tokens each and identical completion IDs:
`/tmp/dsv4_tp8_war_post_A3_long_20260907.json`. This establishes successful
transitions in this run, not broad stress-test coverage or factual correctness
of the generated code-audit claims. Next validation should combine the fallback
fix with the opt-in GEMV candidate in a fresh service; no promotion yet.

## Combined B3 validation

Fresh B3 with POST_REPLAY fallback and GEMV enabled completed the whole C1
harness, all six full-prefix teacher probes and both long-source requests.
Per-task medians83.707 /83.801 /83.998 tok/s. Across nine measured requests,
median83.857 versus A3 76.818 (+9.16%); trimmed mean83.874 versus76.735.
One62.725 tok/s sample remains in the record: untrimmed mean81.541 versus
76.731 (+6.27%). This A3/B3 pair is not a complete fresh ABBA by itself;
earlier B1/B2/A2 runs support the speedup but had incomplete transition tests.

All nine B3 completion sequences match the prior independent B2 candidate.
France passes. All six full-prefix teacher probes match the original control,
but these still do not exercise the decode-only candidate. The candidate
changes two short-task trajectories relative to control; it is not bit-exact
with old GEMM arithmetic. Both long-source responses terminate normally at460
tokens, have identical IDs, and match the first completed B2 long response.
Do not promote its unsupported code-audit claims to verified factual answers.

The adjacent JSON preserves A3/B3 timings, completion hashes and long texts.
The large-prefill P32 regression test is running next. Keep the new selector
opt-in until broader transitions and mixed tiers have been checked.
