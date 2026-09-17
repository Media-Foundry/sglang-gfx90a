# First-MHC table 16K regression passed; fixed QKV solution screen is promising

Continuation of f239f3a794, original V4 TP8/EP1/native AR, original checkpoint,
1M logical KV, 32K chunk. No production source/default changes this turn.

## Completed C16x16K service ABBA

.agents/experiments/dsv4_first_mix_table_lengths_20260917/16k, run session92889
exit0. The sole A/B variable is the first-MHC table. COMMON_FP32/SMALL/UNFUSED,
route producer, wide owner and owner K32 are enabled identically. Sixteen actual
public-code requests,262141 input tokens,warmup excluded,zero cache hits,one
output token per timed request. Source-hash/actual-path/1M-pool guards passed.

- A1 10189.966790 input tok/s
- B1 10234.326131
- B2 10230.282208
- A2 10193.121297
- Control center10191.544044,candidate10232.304169: +0.399941%.
- Control drift0.030957%,within-leg ranges0.052..0.202%.

Rates independently recomputed from earliest request start to last first token.
Three fresh processes x two128-token code waves: all16 outputs identical across
all6 waves. A1/A2 and A1/B teacher forcing both exact at1008 selected logprobs
and Top5 lists. France passed. All owned services stopped,GPUs confirmed empty.
Evidence archive ~5.74MB includes raw responses/timestamps and runtime sources.
This is measured same-workload stability,not universal batch invariance.

32K regression has started with the same harness: session98457. A1 completed
9904.922289 input tok/s (3waves); B/A2 not complete at this checkpoint. Do NOT
claim a32K candidate result from A1. Re-poll the exact session and read driver
logs; observation timeout is not terminal. No concurrent GPU tests during it.

## Bounded new QKV component direction

.agents/experiments/dsv4_qkv_fixed_solution_20260917/oracle.py,session78385 exit0.
Physical GCD5 only,after16K services fully stopped and before32K service began.
Uses actual captured layer0 normalized rows and SHA-verified loaded BF16 QKV
weight[1536,4096]. An explicit per-object rocBLAS handle,atomics disabled,
algo=solution_index. No service/global production BLAS selection changes.

Important API finding: installed rocBLAS5.6 build8d1ae90e returns negative IDs
for Tensile. Initial positive-only guard rejected them before any candidate
execution(session47024 exit1). Historical filenames index-zero* and assertion
text mention zero,but actual returned IDs were NEGATIVE,not zero. That failed
source/results/log is preserved. Do NOT claim it found default index zero.

Pinned implementation at installed build revision:
https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocblas/library/src/include/tensile_host.hpp#L50-L64
and definitions.hpp:28-31 establish0/-1 defaults,positive hipBLASLt,<-10 explicit
Tensile. Public installed header prose is stale. The corrected experiment is
restricted to this build and excludes default/reserved IDs; no ID sign guessing.

520 IDs support all tested M1/64/65/8192/32576. Bounded screen sampled32 spread
across this intersection,NOT exhaustive autotuning. Same preallocated output
and captured5-call graphs; preliminary fastest fixed solution-606601783 takes
3.077205ms atM32576 vs torch3.211066ms (~4.17% less time). This is a screen,
NOT yet a paired ABBA or service speedup. Other finalists-606601866/-606601816
take3.113702/3.121127ms. Fourth-606601918 is slower3.292829ms.

All four finalists pass3 input perturbations x5 M sizes and random row ordering
(60 cases total),and100 changing-input graph replays each(400 exact checks).
They are internally shape/row stable on this fixture; NOT promised bit-identical
to the old automatic BLAS path and NOT yet real late-layer input coverage.

confirm.py is PREPARED,NOT RUN. It validates6 checkpoint weight pairs (layers
0/1/10/20/30/42) and3 mutations x5 M shapes; checks layer0 expansion against the
actual service weight hash; uses captured layer0 inputs for other weights and
labels this limitation. It will measure four graph ABBA rounds on each of
M64/65/8192/32767/32768 for the best3 solutions. Its guard requires32K A1/B/A2
owned-stop proofs; do not run it while the32K service is active.

Next: finish32K service comparison/archive; then run confirm.py in isolation.
If a fixed solution is numerically stable and faster at relevant M,consider a
separate opt-in native-prefill integration/live trace. Do not replace production
based on this single-layer screen,broaden todecode,or claim whole-model drift
solved. Earlier ordinary rocBLAS auto-selection still failed cross-M; earlier
fixed-K Triton geometries were slower,so those are not fresh experiments.
