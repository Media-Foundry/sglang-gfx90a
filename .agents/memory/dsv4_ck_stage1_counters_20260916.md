# Real CK stage1 counters: efficient compute, not a demonstrated LDS problem

2026-09-16, base1224b7ff6c. Historical real original-V4 layer21/rank5 fixture,
M32767/H4096/I256/Top6. Single HIP-visible5 device, PCI0000:b3:00.0; no service
overlap. This is not a fresh current-model capture or eight-rank service timing.

Explicit runtime stage1 kernel:
`moe_ck2stages_gemm1_256x64x64x128_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight0_dsv4silu_B16_B16_B16`.
Entire intermediate equals the captured stage1 output bytewise.100graph replays
also equal. Uninstrumented graph timing median **6.68702ms** (10samples,5calls).
No new arithmetic or service selector changed for this measurement.

Useful gate/up matmul FLOPs =2*M*6*4096*256*2, giving123.315usefulTFLOP/s at
that component time. This excludes padding, SwiGLU/scalar work and data movement;
it is neither a hardware utilization percentage nor a whole-MoE throughput.
Captured valid sorter rows205056 versus196602real assignments:95.877% assignment
occupancy of valid padded rows. Do not count inactive grid-tail CTAs as useful
matrix work or assume stage1 has massive expert padding in this fixture.

Three separate selected-region counter passes; three exact-output calls/pass:

| Metric | Result |
|---|---:|
|FetchSize mean|4781131.67KiB|
|FetchSize per call|5.20448 /4.24931 /4.22514GiB|
|WriteSize mean|97977.67KiB|
|SQ_INSTS_MFMA mean|104988681|
|SQ_INSTS_VALU mean|161276544.33|
|LDSBankConflict derived metric mean|0.332071%|
|Resources|112VGPR,128AccVGPR,16KiB LDS,0scratch|

Fetch is all measured kernel HBM reads, not just weights; it includes activation
and routing/metadata traffic. Variation across calls is retained, not hidden by
the mean. These counts do not prove a single bottleneck or peak-bandwidth use.
LDS bank-conflict evidence is small, so a speculative LDS-layout rewrite is not
the first supported follow-up. No claim that all supply optimization is exhausted.

Profiler durations are diagnostic only: SDK substitutes intercept queues and
does not preserve original priority/CU masks. Version-query child processes
receive uninstrumented environments to avoid the previously documented SDK
stdout contamination; actual command results are used, never fabricated.
Parent GPU profiling is unchanged; GPU_ARCHS=gfx90a is checked against hardware.
All three passes succeeded, each with only3stage1 kernel records.

The follow-up moved to the separate FP4→BF16 preshuffle producer: its source
shows three CTA barriers per512-value tile, whereas the producer can directly
emit the same CK layout. That decision is based on code plus subsequent oracle,
not an assertion that stage1 counters measured the dequantizer.

Artifacts `.agents/experiments/dsv4_ck_stage1_counters_20260916/`: probe.py,
collect.py,analyze.py,timing.json,summary.json,evidence.tar.gz and manifest.
Raw traces, counter CSVs, commands, profiler logs, module/source hashes and exact
replay receipts are retained. No installed AIter module or weight file modified.
