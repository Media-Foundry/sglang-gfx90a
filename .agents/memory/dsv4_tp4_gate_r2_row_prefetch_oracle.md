# TP4 M32 grouped-gate R2 row-prefetch oracle (2026-08-30)

## Scope

This standalone oracle tests a scheduling-only change to the production
gfx90a TP4 grouped gate/up kernel.  It retains A4/R2/W8/G2080, the LDS pair
LUT, packed FP4 weights, INT8 activations, every SDOT call, FP32 accumulation
order and the DPP wave64 reduction tree.  For one K group it requests the
packed `uint4` and E8M0 scale for gate/up row0 and row1 before decoding or
consuming row0.  This is distinct from the rejected K-group distance-one
prefetch and from the accepted grouped-down R2 row-prefetch.

Standalone files, not connected to production:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh`
- `scripts/rocm/bench_dsv4_tp4_gate_row_prefetch_oracle.py`

## Resources and correctness

The gfx90a code object reports 95 VGPR, 51 SGPR, 1 KiB LDS, zero private or
scratch segment and no spills.  On the real diverse pass37/layer34 route
(106 active experts, 192 assignments and 113 A4 scans), 100 eager activation
mutations and 100 captured-graph mutation replays were bitwise exact for the
gate intermediate, INT8 values/scales, down FP32 partial and final BF16 output.

## Seven-round ABBA

Trimmed means in microseconds:

| stage | production | row-prefetch | delta |
|---|---:|---:|---:|
| gate/up | 255.534 | 244.741 | -10.793 |
| quant | 40.709 | 41.225 | +0.516 |
| down | 171.518 | 171.451 | -0.067 |
| reduce | 3.809 | 3.821 | +0.012 |
| full routed | 439.162 | 423.595 | -15.567 |

The isolated gate improves 4.22% and the full routed path improves 3.675%.
It misses the predeclared 10% and 395-us continuation gates, so it is not by
itself a performance checkpoint.  Unlike the K-group prefetch, however, this
schedule is a stable exact small win.  If pursued, the next valid experiment
is to combine it with the already exact grouped-down R2 row-prefetch and run a
teacher-forced plus service ABBA; do not infer end-to-end benefit solely from
this microbenchmark.

## Production-stack service rejection

A strict default-off selector was added for only gfx90a TP4 M32, Top-6,
E256/I512/H4096, A4/R2/W8/G2080 and LDS mode2. It was temporarily combined
with the accepted DPP gate/down-row-prefetch path and issue-order-3 profile.
The candidate matched the adjacent 32-row teacher oracle exactly for output
IDs, complete token logprobs and top-5 rows.

Five diverse 512-token resident runs were
`623.571/624.675/623.896/623.269/623.455 tok/s`, median `623.571` and trimmed
mean `623.640`. The adjacent issue-order-3 baseline is about `623.4 tok/s`, so
the apparent gain is only about 0.04%, below the 0.5% stable-small-win policy.
All rounds passed France-prefix and length checks. Cross-round hashes retained
the known scheduler-path drift, while the fixed-batch teacher oracle was exact.

Remove the TP4 profile opt-in. Keep the selector default-off and the standalone
oracle for documentation, but do not enable it in production: the isolated
10.8-us gate saving is hidden by the full multistream graph.
