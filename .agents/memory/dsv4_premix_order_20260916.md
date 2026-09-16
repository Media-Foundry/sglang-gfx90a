# Pre-mix CTA ordering: exact but small; hardware counters guide next oracle

Accepted service speed remains8964.910269 input tok/s on original V4 TP8 C16x8K,
original checkpoint,1M logical KV. No new service throughput is claimed here.
All GPU work below ran on physicalGCD5 after the TP8 profile service stopped.

## Bounded ordering screen

Copies the current premix8_pair arithmetic verbatim (eight rows,two independent
columns,K1024 chunks,FP32 Fn,reduction order and RMS scaling). Only CTA ordering
changes. Groups1/2/4/8/16/64 change how many row groups are traversed before the
next column pair. Ragged last groups have explicit bounds; no D2H or workspace.

Inputs repeat/scale real sampled layer0 x/residual/post/comb/Fn, not32767 unique
live model rows. Current exact HIP post produces each candidate's pre-mix inputs.
M17/8192/32767,five mutations per candidate; all mixes byte-exact. Three graph
ABBA cycles, five replays/sample. Best large-M group8:

- Isolated M32767:4.973573 ->4.884022ms (~1.80% latency reduction).
- Isolated M8192:1.369369 ->1.371705ms (flat/slightly slower).
- Group64 M32767:4.970036 ->6.701385ms, a clear negative.

Then include the actual producer in every graph replay (exact HIP post followed
by pre-mix), rather than measuring only warmed pre-mix input buffers:

- Full boundary M8192:1.985646 ->1.985646ms.
- Full boundary M32767:7.404877 ->7.306747ms (~0.098ms saving).

Both complete boundaries remain byte-exact. This is a small isolated result,
not an E2E gain. Retain the oracle but do not add a production switch at this
stage; it does not justify continuing an unbounded CTA-order sweep. No current
runtime arithmetic,precision,backend selection or launcher defaults changed.

## Counter evidence on the baseline pre-mix

ROCTX-selected regions,three calls/pass,only premix8_pair recorded. Inputs use
the same historical sampled rows atM32767,not a new live service fixture. SDK
warns that interception uses a system-memory queue and drops priority/CU mask;
profiled times are diagnostic,not an uninstrumented throughput claim.

- Profiled duration:about4.93–4.96ms.
- FetchSize mean2,088,287.96KiB (~1.99GiB); WriteSize11,133.68KiB.
- SQ_INSTS_VALU590,315,520; SQ_INSTS_VMEM20,054,016; identical across repeats.
- Triton reports78 registers; profiler reports80 allocated VGPRs,0scratch,0LDS.
- Profiled outputs repeat exactly.

These counts do not prove VALU utilization, L2 saturation or an HBM physical
limit. They show that the18GiB of logical per-CTA input/Fn requests must not
be equated with18GiB of actual HBM reads. CTA cache reordering is not a large
remaining HBM-byte opportunity in this sample. Instruction/data-reuse changes
are more promising than treating this as another cache-order-only problem.

Profiler child GCC processes also inherited SDK instrumentation during the
first Triton launcher build and emitted SPM warnings. The run still completed;
the CSV contains exactly the three requested GPU dispatches,not those compiler
processes. No child outputs were fabricated or replaced. Raw CSVs are archived.

## Next bounded candidate, not an accepted implementation

The local CDNA2 ISA chapter7 table25 lists FP32 MFMA16x16x4, and AIter's CK
`include/ck/utility/amd_xdlops.hpp` implements its float4 accumulator intrinsic.
Consider directly loading BF16 activations, converting them in registers, and
retaining original FP32 Fn for a narrow FP32 MFMA pre-mix oracle. This differs
from the failed BF16->full-FP32-activation + torch.mm experiment: it must not
allocate the old2GiB activation copy or lower Fn precision.

Start with lane/accumulator mapping on small exact integer matrices, then real
inputs,ragged rows,row permutation and replay. MFMA changes the summation/FMA
order: do NOT call it legacy-bit-exact or silently promote it. Any useful speed
must pass explicit numerical,teacher-forced and service quality gates before
selection; current exact8965 checkpoint stays available. No MFMA prototype or
MFMA speed measurement exists yet at this entry.

Artifacts: `.agents/experiments/dsv4_premix_order_20260916/`. No GPU processes
remain after these tests. Runtime files are unchanged in this follow-up turn.
