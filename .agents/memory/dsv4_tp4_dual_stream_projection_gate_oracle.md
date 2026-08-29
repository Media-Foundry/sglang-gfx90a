# TP4 M32 dual-stream projection-only gate/up oracle (2026-08-30)

## Hypothesis and implementation

Unlike the rejected 1024-thread paired-wave CTA, this oracle launches two
independent 512-thread kernels on two HIP streams.  One computes only the gate
projection and one computes only up.  Both retain production A4/R2/W8/G2080,
the wave64 K traversal and shuffle tree, packed FP4 weight/scales, LDS pair LUT
and SDOT arithmetic.  Each writes its reduced projection to FP32; after stream
join, a small 384-CTA kernel performs the original clamp, SiLU, multiply and
BF16 store.  The ordinary quant, down and fixed reduction follow unchanged.

The path is component-only and is not wired to production:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_gate_up_dual_stream_oracle.cuh`
- `scripts/rocm/bench_dsv4_tp4_m32_dual_stream_projection_oracle.py`

The benchmark captures both baseline and fork/join candidate graphs and uses
the real diverse-request pass37/layer34 route (106 active experts, 113 A4
scans).

## Static resources

Natural projection-only code objects are still:

- gate/up projection: 68 VGPR, 54 SGPR, 1 KiB LDS, zero spill;
- combine: 18 VGPR, 16 SGPR, zero LDS/spill.

Replacing separate token/slot/valid arrays with one encoded-ID array did not
lower the allocator below 68 VGPR.  `launch_bounds(512,2)` also did not impose
a gfx90a VGPR cap.  An explicit `amdgpu_waves_per_eu(4,4)` variant was tested
to force the desired residency; it made producer/full-graph latency about
393/559 us and was removed.  Thus the requested `<=64 VGPR` static gate was
not reached without changing A4/R2 arithmetic or forcing an even slower
schedule.

## Correctness

The retained natural variant passed:

- 100 independently mutated eager executions;
- 100 graph replays with mutated activation and router weights.

Intermediate BF16, group-32 INT8 value/scale, FP32 down partial and final BF16
were bitwise equal to the production grouped gate/up path in every case.

## Graph ABBA result

Seven rounds, 30 iterations/sample on GCD0:

| stage | production | dual stream |
|---|---:|---:|
| producer | 255.838 us | 349.692 us |
| quant | 40.408 us | 40.807 us |
| down | 172.088 us | 172.260 us |
| reduce | 3.727 us | 3.860 us |
| eager full | 439.717 us | 529.369 us |
| captured graph full | 436.336 us | 516.071 us |

The graph candidate is **15.45% slower**, far above the `395 us` continuation
gate.  Even if 68 VGPR permitted partial overlap, duplicating 2080 CTA LUT
initializations, metadata/activation traffic, writing and rereading two
`[32,6,512]` FP32 tensors, the stream join and combine kernel dominate the
register-pressure benefit.  Do not connect this representation to production
or continue its geometry sweep.

