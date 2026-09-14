# C16 residual drift: isolate wq_b after the QKV repair

Base commit: `5b212ae632`. Original V4-Flash, TP8/EP1, native AR, original
checkpoint, 1M logical KV pool. The prior turn repaired one projection source
of drift but did not establish whole-model determinism.

## Frozen service-input component screen

`projection_library_screen.py` uses the real layer0 normalized q_lora rows
from `layer0-stable-qkv`, their original row positions, and each of the eight
real TP shards of `layers.0.attn.wq_b`. Weight dequantization follows the
checkpoint's block128 FP8/E8M0-to-BF16 contract already used by the service.
All component GPU work uses only physical GCD4; eight "ranks" in this screen
mean eight different real rank weight/input fixtures, not eight GPU processes.

For every fixture, `F.linear` reproduces both service snapshots exactly.
Changing request placement changes54 sampled output elements across the eight
fixtures. Four alternatives have zero shifted-row differences on the fixture.

| Implementation | Median component time across fixtures/shapes | Shift differences |
| --- | ---: | ---: |
| Existing F.linear | 2.2452 ms | 54 elements |
| Fixed128/128/128, eight waves | 2.7319 ms | 0 |
| Fixed64/128/128, four waves | 4.7054 ms | 0 |
| Transposed GEMM plus contiguous copy | 8.3795 ms | 0 |
| Fixed128/256/64, eight waves | 3.0994 ms | 0 |
| Fixed64/256/128, eight waves | rejected: 80 KiB LDS exceeds64 KiB | not executed |

The first screen stopped at the LDS rejection. Its original log is retained;
the revised driver records this expected candidate failure and continues.
This is not an OOM or a failed service. No external task was terminated.

The separately integrated fixed128 service entry matches the offline kernel.
For each of the eight fixtures,100 real-input perturbations at varied row
offsets pass (800 total). This is row-placement invariance, **not** bitwise
parity with the old GEMM. The fixed tile has different accumulation order and
is slower; no throughput win or production-default change is claimed.

## Narrow service wiring

`SGLANG_DSV4_DEBUG_PREFILL_WQB_STABLE` defaults OFF. The ordinary original-V4
prepare path applies the same native/HIP/gfx90a/TP8/EP1/M8192..36864 predicate
as the existing diagnostic selectors. The private `_compute_q_b` argument
defaults false; decode, draft, other model families and alternate prepare
paths do not opt in. Biased or unexpected weight layouts fail explicitly.

Seven CPU contract tests pass, including a source-level check of the actual
normal-prepare callsite and disabled/no-GPU behavior. The service run is
`layer0-stable-qkv-wqb`, changing only wq_b relative to `layer0-stable-qkv`.
It retains exact token echoes, fresh cache salts, complete KV snapshots,
153 evidence-selected sample positions and same-order control replay.

Service findings will be appended only after that run completes.

## Completed service result

The run completed warmup/A1/B1/A2, all16 exact input echoes each wave, and
stopped its owned server. Eight stable-wq_b hits were logged. Full corresponding
QKV/KV snapshots remain exact, including all eight ranks' complete KV histories.
At all retained sample positions, Q-before-RoPE, Q-after-RoPE, attention-core,
inverse-RoPE and stable wo_a are now exact across the placement change.

The first remaining divergence is **wo_b partial**:32 rank/row pairs differ,
although their wo_a inputs match. The replicated attention output differs at
five positions (768,4864,5632,6144,7424), each in one BF16 element. FFN inputs
differ at three positions. FFN output differs at10 sampled positions, versus
16 with QKV-only repair and147 before either query repair. Independently, the
shared-expert path still changes on some rows whose FFN input is equal.

Same-order A1/A2 sampled stages are all exact, and all16 complete128-token
outputs match in this run. A1/B1 complete outputs match13/16. One successful
same-order pair is not a guarantee across launches, earlier prefill groups,
arbitrary lengths or batch compositions. Traces still retain only the final
prefill group. Do not claim whole-model bitwise stability.

This establishes a second service-validated partial repair. It does not add
a production performance default or an accepted C16 speed result. The
remaining projection effects are consistent with the already demonstrated
row-dependent GEMM rounding mechanism, but wo_b/shared need their own frozen
input tests before calling their causes proven.

## Performance follow-up, not yet measured

The existing M36864 large-CK bound originates in the TP4 OOM experiment
`dsv4_gfx90a_bf16_ck_batch_ceiling_20260902.md`. Both the current runner selector
and helper enforce it for TP8 as well. A TP8-only M65536 experiment could
reduce this C16 x8K workload from four large forwards to two, but requires
checking sorter encodings, workspace/1M-KV capacity and actual end-to-end
behavior. No speed or capacity claim is made from this source inspection.

The actual AIter tree is `/home/pc/pytorch/third_party/aiter`, not the stale
`/home/pc/Code/aiter` assumption. Its CK sorting kernel encodes token IDs in24
bits and Top-K slots in8 bits (`moe_sorting_kernel.hpp`), and its GEMM consumers
mask token IDs with0xffffff. This removes a suspected16-bit token-ID barrier
to M65536, but is not an execution/capacity test. The runtime helper and runner
remain unchanged at M36864 in this commit.

## Evidence bundle

`wqb-evidence.tar.gz`:36 files,4,176,515 bytes, SHA256
`b2ec15f2452b407f3909b59e1a800942f1d83bed9f54a812c51054fdb6f64842`.
The manifest records local activation hashes; full activation/weight tensors
are excluded. `validate_wqb.py` checks only the bounded claims above. All owned
GPU tests and the service exited; unrelated report work and graph-memory
pickle changes were not part of this experiment.
