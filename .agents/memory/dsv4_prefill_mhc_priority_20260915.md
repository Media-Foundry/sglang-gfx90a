# Large-prefill MHC priority screen: reject speed change; isolate iteration switch

This follows the completed C16x32K query-only ABBA (+20.35%, separate record).
All TP8 services were stopped before testing, and all GPU work here used only
physical GPU4. No production numerical path or launcher default was changed.

## What was compared

The current single-request large-prefill boundary reaches legacy split-K
with cached FP16 Fn and8 Sinkhorn iterations. Its batch-size admission guard
preempts the configured FP32 pre-mix8 path.

- A: current batch1 full boundary, FP16 Fn /8 iterations.
- B: suppress legacy admission only in active large-prefill scope; existing
  FP32 pre-mix8 and batch1 /8 iterations.
- R: existing batch2 FP32 path, **20 iterations**.
- R8: R with only the Sinkhorn batch hint aligned to1, hence8 iterations.

B/R8 must be byte-exact. R20 is retained as a diagnostic, not incorrectly
treated as the same mathematical reference. Small M keeps the legacy
batch1 path in every arm. The single-GCD harness disables symmetric
allocation and supplies a None TP accessor, following existing MHC oracles;
all temporary overrides are restored in `finally`.

## Completed results (allocating full boundary, not service throughput)

| M | A median ms | B median ms | A/B |
| ---: | ---: | ---: | ---: |
| 1 | 0.217888 | 0.219440 | 0.992927 |
| 8192 | 3.018479 | 3.320943 | 0.908922 |
| 32767 | 11.929380 | 13.264468 | 0.899349 |
| 32768 | 11.965525 | 13.280068 | 0.901014 |

Each shape uses three component ABBA cycles, five calls per event sample.
Raw samples are retained, including a13.450ms A sample at M32768.
No graph timing or eight-rank critical-path speed is claimed. At large M,
the full candidate boundary takes about10–11% longer: **do not wire this
priority replacement into a service speed trial**. This does not invalidate
the previously accepted multi-request FP32 reuse4->reuse8 optimization.

M1/M8192 each have3 mutation checks, M32767/M32768 each10. B/R8 outputs are
byte-exact for every checked mutation; all output tensors finite; candidate
row permutations exact; A and B each6/6 same-input eager replays exact per
shape. M1 A/B outputs are unchanged. These are finite component checks,
not a proof of arbitrary-input or full-model determinism.

## New numerical finding: iterations depend on request batch size

`mhc.py::hc_split_sinkhorn` uses the environment override for batch1 and20
otherwise. The fused split-K tail uses the override directly. The actual
launcher override is8; `/home/pc/models/modelscope/config.json` specifies
`hc_sinkhorn_iters=20` (config SHA256
`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`).

For the initial fixture, R8 versus R20 differs only in returned comb, with
max_abs **0.16722455620765686**. Residual, post and current layer input are
exact; comb affects later HC transitions. Aligning the iteration choice
removes B/R8 differences completely. The A8/B8 initial precision-path
differences are much smaller: post max_abs1.80006e-5, comb6.32107e-5,
BF16 layer input0.00390625 (mutated inputs up to0.0078125).

Important fixture scope: real layer0 rank0 residual/Fn/scales/base/norm are
loaded from the prior trace, but residual rows are repeated/sliced and the
post input starts at zero with identity combine, then is synthetically
perturbed. The0.167 number is **not a fresh complete service-boundary trace**
and not a measured final-logit error. Repeating the same residual across M
does not create independent evidence from multiple layers.

The batch-dependent iteration/precision policy can explain some changes
when admission changes batch shape. It cannot alone explain32K control
drift when both executions retain batch1/8 iterations. Previously isolated
MoE FP32 atomics and other shape-sensitive projections remain relevant.
The next correctness experiment should isolate an explicit large-prefill
iteration policy with fixed inputs, without changing C1 AR/DSpark or silently
combining precision changes with performance gains.

## Harness failures and artifacts

Attempt1 (session57456) passed M1, failed M8192 at an uninitialized TP-group
accessor. Attempt2 (60043) passed M1, then failed the unmatched B8/R20
equality assertion. Neither provides accepted large-M timing; their partial
JSONs retain status `running` but their processes exited1. v3 sessions59093
and61617 completed successfully. Do not interpret stale JSON status as a
live process.

Directory `.agents/experiments/dsv4_prefill_mhc_priority_20260915/` contains
the driver, four CPU contract tests, preserved failure record/source archive,
and both successful JSONs. Failed v2 source archive SHA256:
`71db3a3cc1c98751b35a3886c8188bb6773ba2ca3a05f27ab26165220a7a5522`.

Successful JSON hashes:

```
207c7eabfa828c49341fb6f3914df47ad5de90b170e9ab05d13e119fdb02bcc5  screen-small-v3.json
d684b81c3d074a10bd64abe00818335dae5450714db9fe0d503d147d63925500  screen-large-v3.json
```

Driver and captured tensor hashes are in the successful result headers;
the original local tensor prerequisites are not copied into the repo archive.
After both screens, `amd-smi process --json` again reported no processes on
any GCD.
