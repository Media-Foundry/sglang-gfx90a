# TP8 M32 gate uniform metadata: component rejection

Baseline commit `96e8267177`. Native TP8 service PID3391501 retained its
1,048,576-token pool, checkpoint precision and production kernels throughout.
No service restart or new production selector. AMD-SMI audit before the first
run found only the service tree; the second run audit included the temporary
oracle PID3413086, which exited normally. Both tests used physical GPU4.

## Hypothesis and exact scope

The accepted A4/R2/W8/G832 prefetched gate assigns a single task to each wave.
Its expert ID and four encoded token/slot IDs are identical across all lanes.
An oracle-only `SGLANG_FP4_GATE_UNIFORM_METADATA_ORACLE` applies
`__builtin_amdgcn_readfirstlane` to those values to expose uniformity. SDOT,
K traversal, scale layout, reduction, output address and task order are unchanged.
The macro is absent from production compilation. Existing source without the
macro retains its original behavior.

Unlike earlier expert-row persistence and row-stripe experiments, this does
not reschedule expert runs, add a cache, expand accumulators or add barriers.
All-memory search found no previous readfirstlane metadata experiment.

## Static and timing results

ROCm7.14 llvm-objdump offloading extraction and llvm-readobj notes from the
fresh paired modules show W8 VGPR95 ->76, SGPR51 ->68, LDS1024 bytes unchanged,
private segment0 and no SGPR/VGPR spills in either. Reduced VGPR alone does not
establish higher achieved occupancy or improved latency; no hardware counters
were collected to attribute the slowdown.

Five ABBA cycles, ten samples per arm, discard each arm's extrema:

| Expert pool / active / A4 scans | W8 baseline / uniform (us) | W4 paired baseline / uniform (us) |
| --- | ---: | ---: |
|256 /133 /133|275.576 /279.044|275.369 /276.170|
|128 /104 /105|240.949 /249.022|242.342 /246.682|
|32 /32 /61|183.043 /185.791|183.445 /183.211|

W4 uses1664 CTAs to preserve6656 total grid waves, retaining A4/R2; it is a
targeted follow-up to the lower VGPR count, not a broad geometry sweep.
W8 regresses1.26–3.35%; W4 regresses0.29–1.79% on the two diffuse routes,
and improves only0.13% on the concentrated route. Neither merits E2E promotion.
Do not infer that all uniform metadata transformations are bad, but do not
repeat this full expert+token transformation or claim the VGPR reduction as
a performance win.

## Correctness and measurement limits

Each of six cases passed100 activation/scale mutations with exact torch.equal
comparisons of gate BF16 intermediate, down FP32 partial and final BF16 output;
maximum final absolute error0. Per mutation, ten additional candidate graph
replays remained equal (1000 total). This is tensor equality, not an explicit
raw-bit signed-zero check. Routing is fixed within each case and synthetic;
it is not a captured real model input. Weight scale values mutate, rather than
remaining constant. No claim of new model E2E correctness is made.

Timed chain is gate + intermediate INT8 quant + down partial + fixed reducer.
Sorter and initial input quantization are outside timing, despite the inherited
script field `full_stage=true`. Samples use100 graph replays per event interval,
five ABBA cycles, same allocations; not a cold-weight full-model benchmark.
All raw paired samples are in the adjacent JSON. C1 is not exercised.

Commands (DS Python, repo python on PYTHONPATH, HIP_VISIBLE_DEVICES=4):

```sh
python scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --uniform-metadata
python scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --uniform-metadata --uniform-waves 4
```

Decision: preserve the standalone reproduction, keep current production path.
No new E2E speed claim; no persistent weight/workspace allocation or KV reduction.
