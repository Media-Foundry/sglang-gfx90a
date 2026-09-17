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

32K regression subsequently completed: session98457 exit0,all services stopped.
524286 input tokens/wave,same guards and three waves/leg:
A1/B1/B2/A2=9904.922289/9939.825955/9938.911974/9894.194662.
Control9899.558476,candidate9939.368965,+0.402144%. Control drift-0.108306%,
within-leg ranges0.023..0.076%. All16 outputs identical across6 waves; both
teacher comparisons exact at1008 logprobs and Top5 lists. Evidence archive
10.22MB includes raw responses/timing/source/owned-stop proofs. amd-smi then
confirmed all8 GCDs empty before the next component run. No overlapping jobs.

Same measured combination now has8K10377.622623,16K10232.304169,32K9939.368965.
Lengths/requests are genuine code fixtures,not repeated random tokens. These
are independent length-specific ABBA results,not per-user decode throughput.

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

## Formal confirmation completed; common M32768 is NOT a speed win

First fresh-process confirm(session68881) failed on the first explicit-index
call with rocBLAS internal_error6,before any numeric/timing case. Layer0 weight
expansion already matched the captured live-weight SHA. Failure report/log and
that source are preserved in confirm-no-enumeration-evidence.tar.gz.

Retry(session10929 exit0) enumerates supported solutions for actual shapes in
the fresh process BEFORE invoking indices,and verifies membership. It succeeds.
This is consistent with lazy solution-library/index-map initialization; do not
claim a complete upstream root-cause proof. Production must not blindly reuse
bare IDs; installed-version/library-hash and per-process preflight matter.

Six checkpoint weight pairs(layers0/1/10/20/30/42),three mutations,five M shapes,
three algorithms: all270 cross-M/random-row-order cases pass. Inputs for later
weights remain captured layer0 samples,NOT actual late-layer activations.

Paired4-round graph ABBA for best solution-606601783 (ms,preallocated output):

| M | Control | Fixed solution | Time reduction |
| ---: | ---: | ---: | ---: |
|64|0.039921|0.071489|-79.08%|
|65|0.039121|0.071777|-83.48%|
|8192|0.810269|0.775260|+4.32%|
|32767|3.214531|3.081713|+4.13%|
|32768|3.086977|3.086049|+0.030%|

Thus the common service M32768 is effectively tied,not4% faster. Other finalist
algorithms are1.06..1.11% slower at32768. Fixed outputs differ from old BLAS
(14 elements atM64,max_abs0.00048828125;5760 atM32768,max_abs0.0009765625).
Internal shape stability is NOT old-reference bitwise equivalence.

All component jobs exited and amd-smi confirmed no GPU processes. Production
and global defaults unchanged. Confirmation evidence is archived separately;
the earlier screen archive truthfully still labels confirmation as then-pending.

Next: use the fixed algorithm only as a separately scoped correctness/live-trace
candidate if addressing the earliest QKV drift. It is not a universal speed win;
small-tail cost and real full-stage/service timing must be included. For main
speed work,do not extrapolate irregular-M gains to current32K-aligned waves.
Earlier ordinary rocBLAS auto-selection still failed cross-M,and earlier fixed-K
Triton geometries were slower. Whole-model batch invariance remains unresolved.
