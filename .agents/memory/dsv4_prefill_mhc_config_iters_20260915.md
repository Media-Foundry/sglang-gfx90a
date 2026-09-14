# Opt-in native TP8 prefill config20 Sinkhorn policy — component accepted, service pending

Preceding evidence: `dsv4_prefill_mhc_priority_20260915.md` isolated a
batch-dependent8/20 iteration switch. The slower FP32-boundary priority
replacement was rejected. This experiment changes only iteration selection,
not Fn precision, MHC geometry, sorting, attention selection or checkpoint.

New env `SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS=1` is **default-off**. It binds
the local model config's20 iterations throughout original-V4 TP8 ordinary
eager EXTEND, including M1/128 and irregular tails. It excludes all decode,
mixed P/D batches, speculation/draft, TP4, EP/CP/DCP/PP splits, V4.1,
rewritten/TBO batches and CUDA graph capture. Existing profile defaults are
not changed. No service correctness/performance result is established yet.

The context is applied at `_execute_extend`, restored in `finally`, and
consumed by split-K fused tail, native standalone Sinkhorn, and native
post/pre/finish iteration selectors. Disabled decoration returns the
original callable. Consumers retain their legacy defaults outside the scope.

## Completed component oracle on physical GPU4

A: env8/no scope. B: same env8, scoped config20. R: existing global env20.
Batch1 keeps FP16 Fn and the same fused-tail kernel family in all arms.
Batch2 keeps its existing FP32 boundary. Tests use a repeated captured
layer0 residual/Fn with synthetic post/combine, not a full fresh service trace.
All TP8 services had stopped before the GPU checks.

| M | request batch | A median ms | B median ms |
| ---: | ---: | ---: | ---: |
| 1 | 1 | .224225 | .223761 |
| 128 | 1 | .219825 | .222161 |
| 128 | 2 | .324593 | .322545 |
| 8192 | 1 | 3.022027 | 3.236108 |
| 8192 | 2 | 3.372588 | 3.369516 |
| 32767 | 1 | 11.943439 | 12.823507 |
| 32767 | 2 | 13.395750 | 13.398038 |
| 32768 | 1 | 11.963984 | 12.845731 |
| 32768 | 2 | 13.379030 | 13.379190 |

These are allocating full-boundary component timings from three ABBA cycles,
five calls per event, not E2E or graph throughput. Additional20-iteration
work costs about7.1–7.4% on batch1 large M; already20 batch2 is equivalent.
This is a correctness tradeoff, not a claimed speed gain.

B/R full outputs are byte-exact for all65 mutation checks (25 small-screen,
40 large-screen). Each of nine shape/batch cases has10/10 same-input eager
replays exact for A and B, finite candidate output checks and exact row
permutation. Batch2 A/B outputs are unchanged. On the initial batch1 fixture,
only returned comb changes (max_abs up to .1672292054), not current normalized
input. This is not proof of global determinism or overall answer accuracy.

31 CPU tests passed across new policy and existing MHC scope/logging tests;
one preexisting pytest unknown `asyncio_mode` warning. These include ordinary
tail admission, explicit mode/architecture/parallelism exclusions, exception
and nested-scope restoration, and compiled-argument selector checks.

## Service experiment in progress

Directory `.agents/experiments/dsv4_prefill_mhc_config_iters_20260915/`:
`oracle.py`, `screen.json`, `large.json`, `run.py`, `sweep.py`, `status.py`.
The service ABBA keeps wide-query16 on in all arms, original checkpoint,
native AR decode,1M pool and32K budget. The sole flag change is config20
prefill policy. Each leg has three one-token timing waves; each fresh process
also has France and two128-token quality waves with full input echoes.
Actual policy hits and unchanged FP16 Fn must be witnessed on every rank.

Partial readback on September15 (not a completed ABBA): A1 finished with
three-wave median5542.583721 input tok/s. Candidate B1/B2 finished with
three-wave medians5480.998651/5480.220425 input tok/s (about1.11% lower).
B completed and stopped; final control A2 is loading. Do not report this
as an accepted result.
All eight candidate ranks logged legacy8 -> config20 policy selection;
candidate France returned Paris. All32 candidate128-token excerpts have
been manually read: coherent and code-topic-related, without obvious collapse,
not a factual-accuracy certificate. Candidate input echoes32/32 match;
full-answer repeat9/16 and first-token repeat16/16. Seven answers still diverge
later, so config20 is not a complete end-to-end determinism fix. No first-wave
candidate answer fully matched either A1 control wave (first tokens7/16,6/16).
A1 full input echoes passed32/32, but only6/16 full128-token answers repeated
between its quality waves (15/16 first tokens matched). Thus control itself
is not deterministic. Manual control inspection found coherent text but
also unsupported assertions, so it is not an answer-accuracy oracle.

Do not promote the flag until this service comparison, output review and
cold-shape checks finish. Input equality does not imply repeated output;
FP32 MoE atomic reduction and other batch-sensitive paths are still present.
