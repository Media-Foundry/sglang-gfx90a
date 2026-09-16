# Fixed-order vec4 reducer: small exact C16 prefill improvement

Original V4 / TP8 / EP1 / native AR / original checkpoint / 1M logical KV.
This is a bounded follow-up to the real CK counters in
`dsv4_prefill_extras_rebuild_20260916.md`, not another change to GEMM arithmetic.

## Implementation and component evidence

Each thread handles four adjacent output columns. It loads FP32 vectors from
the same six assignment slots, adds slots0..5 in exactly the existing order,
then rounds once to BF16 (or emits FP32 in the oracle). Grid1664 x256 threads.
No new workspace, atomics, task queue, cross-CTA synchronization or precision
change. The old scalar reducer remains available.

The experimental service switch is `SGLANG_DSV4_DEBUG_CK_REDUCE_VEC4=1`.
Default0. It is consumed only inside the verified unique-slot CK path, whose
guard already requires original-V4 TP8 native large prefill. It cannot select
AR decode, draft, target verification, or small-M kernels. The packaged builder
and both private CK/IPC binaries are identical in the service A/B comparison.

Synthetic component screen covers M1,17,8192,32767,36864; BF16 and FP32 output;
three input mutations per shape/dtype;100 graph replays per shape/dtype with
periodic input mutation and poisoned output. All byte-exact to scalar. The
integrated in-tree kernel repeated this gate successfully.

Integrated reducer-only ABBA, BF16 output:

| M | Scalar ms | Vec4 ms | A/B |
| --- | ---: | ---: | ---: |
|8192|0.97802|0.92453|1.05786|
|32767|3.97801|3.75157|1.06036|
|36864|4.48657|4.23539|1.05930|

Historical real CK fixtures M8192/M32767 additionally pass10 inter/route-weight
mutations per shape, both FP32 and BF16 reducers exact. Complete stage2 including
remap, partial allocation, unique CK and reducer (five ABBA cycles):
2.88184 ->2.82675ms and10.84942 ->10.63703ms respectively. This is approximately
2% stage2 improvement, NOT 6% complete MoE or service improvement.

Inspected JIT object's dependency hash matches the measured header. Each vec4
loop has six `global_load_dwordx4`; BF16 writes `global_store_dwordx2`, FP32 writes
`global_store_dwordx4`. gfx90a wave64,32VGPR,0AGPR,0private scratch/spills.
These are decoded compiled instructions/resource metadata, not live counters.

## Service ABBA

C16x8K real distinct code requests,131069 total input tokens,zero prefix hits,
one output token,chunk32768,1048576 pool. Three warm waves per leg, separate
A1/B/A2 processes, adjacent B1/B2 in the candidate process. Common sources and
input manifests frozen. H16, corrected sinks, unique CK common to all arms.

| Leg | Median input tok/s |
| --- | ---: |
|A1 scalar|8662.779814|
|B1 vec4|8698.066455|
|B2 vec4|8682.447714|
|A2 scalar|8666.895835|

Control center8664.837825; candidate8690.257085; approximately+0.293%.
This is a small single-ABBA service gain, not a claim of a large structural
breakthrough. Keep the opt-in scoped path; do not change global defaults or
claim arbitrary workload improvement. All16 requests'128-token continuations
are byte-identical across all12 waves / three processes / both arms (192
responses,24576 tokens). France passes in every process; all services stopped.

## Harness issues and provenance

The first standalone compilation failed because the external experiment header
did not include the JIT csrc search directory. Adding that include path fixed
compilation; no GPU result existed for the failed attempt.

B finished all timing and four quality waves, then its final log-hit assertion
failed: it expected `[TPn]` whereas logging emits `[timestamp TPn]`. The service
was stopped normally in finally. All eight vec4 hits are actually in the log.
`recover_B.py` independently checks those hits, frozen runtime hashes,France,
1M pool,all measured timings,zero cache hits,request IDs,prompt echoes,128 output
tokens per response and decoded text before writing an explicitly labelled
recovered completion record. Original observations are not modified.

The measured harness is preserved as `service_measured.py`; the runnable
`service.py` fixes only that final log-prefix matcher after ABBA. Evidence
archive retains the exact measured source. The previous1312 H16 attention
comparisons belong to the rebuild experiment; no new1312 comparisons are
claimed in this reducer-only service run.

Artifacts: `.agents/experiments/dsv4_ck_reduce_vector_20260916/`.
No universal whole-model batch/logit invariance is established by these finite
same-batch continuation tests. The next larger opportunities still require
new dataflow/counter evidence, not extrapolation of this small reduction win.
