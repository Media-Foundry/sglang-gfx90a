# DSV4 AIter M32 sorter no-clear static experiment

Date: 2026-08-30

## Resolved launch structure

For M32, E256, top-6 and A4, AIter selects CK's small multi-phase path. Torch
profiling observes exactly two kernels per sorter call:

* `MoeSortingMultiPhaseKernel_P0_v2`: about 4.10 us;
* `MoeSortingMultiPhaseKernel_P23`: about 4.57 us.

The P23 grid is `num_experts + get_num_cu()*2`, i.e. `256 + 208 = 464`
CTAs on one MI250 GCD. CTAs `[0,255]` perform sorting; CTAs `[256,463]`
only clear the `[32,4096]` BF16 `moe_buf`. A sentinel initialized to 7.0 is
fully zero after sorting, proving that the clear is real and fused into P23,
not a separate kernel launch. Twenty calls averaged 8.66 us total GPU time.

## Default-compatible static implementation

The default API and its clear behavior remain unchanged. A dedicated
`moe_sorting_fwd_no_clear` API passes `p_moe_buf=nullptr` and zero buffer
dimensions. CK GridSize now detects the null pointer:

* single-phase fallback: clear grid becomes one sorter CTA;
* small P23 path: grid becomes exactly `num_experts` (464 -> 256 here).

The P23 operator therefore never enters its clear branch for the no-clear
call, making nullptr safe. SGLang exposes this only behind the default-off
`SGLANG_DSV4_GFX90A_AITER_SORT_NO_CLEAR` switch and only at the existing direct
grouped FP4 sorting site. The grouped kernels do not consume `moe_buf`.

Static validation completed with Python bytecode compilation and both
repositories' `git diff --check`. No service or performance run was performed
at this stage, and no existing unrelated AIter modifications were changed.

## Compiled micro result

The rebuilt module was tested on the real recorded M32/L34 route. The previous
binary remains recoverable as
`aiter/jit/module_moe_sorting.so.pre-no-clear-20260830`.

Correctness passed:

* 100 randomized route/weight mutations;
* all valid sorted IDs, weights, expert IDs, and `num_valid` bitwise exact;
* 1000 graph replays bitwise exact.

Seven-round A/B/B/A graph timing (500 replays/sample):

| profile | trimmed us |
|---|---:|
| default clear | 10.148 |
| no-clear | 10.029 |

The saving is only **0.118 us (1.18%)**. Removing 208 clear-only CTAs does not
materially shorten the P23 critical path at this small shape; the 256 sorting
CTAs and launch/synchronization floor dominate. This is correct but well below
the threshold for a production performance checkpoint. The experimental
SGLang/AIter/CK implementation was therefore removed with targeted patches;
only the standalone trace, microbenchmark and this record remain. Do not claim
an end-to-end gain or re-add the external AIter ABI for this result.

Micro: `scripts/rocm/bench_dsv4_m32_aiter_sorter_no_clear.py`.
Raw log: `/tmp/dsv4_m32_aiter_sorter_no_clear.log`.

## Temporarily modified and then restored

AIter:

* `3rdparty/composable_kernel/include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`
* `csrc/include/moe_sorting.h`
* `csrc/py_itfs_ck/moe_sorting_kernels.cu`
* `csrc/include/rocm_ops.hpp`
* `aiter/ops/moe_sorting.py`
* `aiter/fused_moe.py`

SGLang:

* `python/sglang/srt/environ.py`
* `python/sglang/srt/layers/moe/moe_runner/aiter.py`
* `scripts/rocm/trace_aiter_m32_sorter_clear.py`

The first eight code paths above were restored to their pre-experiment state.
The two standalone scripts are the retained reproduction artifacts.
