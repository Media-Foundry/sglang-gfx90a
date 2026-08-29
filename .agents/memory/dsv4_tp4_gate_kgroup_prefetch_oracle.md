# TP4 M32 gate/up K-group distance-1 prefetch oracle (2026-08-30)

## Design

For fixed K4096/group32, each wave64 lane consumes two groups: `g=lane` and
`g+64`.  The oracle issues packed dwordx4 loads and E8M0 scale loads for
gate/up row0/row1 of `g+64` before decoding or consuming `g`.  It does not
prefetch activation.  Both groups retain the production SDOT calls, FP32
addition order, A4/R2/W8/G2080 mapping, LDS pair LUT and wave64 shuffle tree.

Standalone files are not connected to production:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_gate_up_kprefetch_oracle.cuh`
- `scripts/rocm/bench_dsv4_tp4_m32_kprefetch_oracle.py`

## HSACO and occupancy

The emitted gfx90a descriptor reports 96 VGPR, 49 SGPR, 1 KiB LDS and zero
spill/private segment.  The adjacent production object reports 94 VGPR,
51 SGPR and 1 KiB LDS.  With four-VGPR allocation granularity both occupy the
same 96-VGPR allocation tier; for a 512-thread CTA this remains approximately
two resident waves per SIMD / one such CTA per CU rather than opening a new
residency tier.

ISA inspection confirms the explicit prefetch survived optimization:

- four adjacent `global_load_dwordx4` instructions hold the second group's
  gate/up row0/row1 vectors;
- the following wait begins at `vmcnt(9)` and gradually consumes outstanding
  operations rather than immediately waiting to zero;
- later packed loads remain vectorized dwordx4;
- no scratch traffic or scalarized packed-weight loads were introduced.

## Correctness and ABBA

Using the real diverse pass37/layer34 route (106 active experts, 113 A4 scans),
the candidate passed 100 eager input/router mutations and 100 graph mutation
replays. Intermediate BF16, INT8 value/scale, FP32 partial and final BF16 were
bitwise exact.

Seven-round, 30-iteration ABBA on GCD0:

| stage | production | prefetch |
|---|---:|---:|
| gate/up | 257.694 us | 265.531 us |
| quant | 39.688 us | 40.600 us |
| down | 172.572 us | 172.779 us |
| reduce | 5.171 us | 5.075 us |
| full routed | 441.592 us | 447.692 us |

The candidate is **1.36% slower** and misses both `gate<=230 us` and
`full<=395 us`.  Explicit distance-one VMEM overlap is real, but the extra
20-ish live packed/scale registers and longer dependency schedule do not add
occupancy; they slightly increase gate latency.  Do not connect this prefetch
schedule to production or extend its sweep.

