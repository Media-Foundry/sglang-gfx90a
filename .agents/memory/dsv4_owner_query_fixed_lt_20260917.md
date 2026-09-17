# New promising exact-to-current owner-query projection: explicit hipBLASLt

Base0ad9862de7. Goal remains original V4 TP8/C16 native prefill,original weights,
1M logical KV. No production source/default changed in this turn; NO new service
speed. Last accepted8K/16K/32K remain10377.622623/10232.304169/9939.368965 tok/s.

## Why this differs from the older owner producer

Old explicit rocBLAS auto-selection compacted Q quickly but changed current
hipBLASLt arithmetic and some TopK membership. It remains opt-in/default-off.
The recent fixed-QKV experiment established explicit algorithm enumeration but
was not a mainM32768 speed win. Here the target is indexer wq_b[8192,1024],not
main QKV[1536,4096]: avoid computing full-M Q on every TP rank while retaining
the CURRENT full producer's values. Small weights projection remains full-M;
compressor/cache must still update all rows. Do not conflate these two GEMMs.

Experiment: .agents/experiments/dsv4_owner_query_phase_20260917/.
Actual historical layer20 q_lora captured atM32767,with full file/input SHA and
original checkpoint-derived BF16 weight SHA verified against the live capture.
This is not a fresh corrected-model input or multi-layer live proof.

## Bounded fixed-Lt / row-phase screen

Isolated process forces ROCBLAS_USE_HIPBLASLT=1 only for the experimental
enumeration shim.733 explicit Lt indices support full/compact tested shapes;
48 spread indices screened. Tested original query16 owners and alternative
whole32/64/128/256-row tile ownership preserving global row-within-tile phase.
Changing ownership alone did NOT establish equivalence. Original owner plan
already works with selected algorithms,so do NOT rewrite the planner yet.

Two indices3778/3953 produce byte-identical current full-M raw BF16 values on
all active rows for all8 owners;3953 is faster(~0.919ms projection only).
These are rocBLAS-visible positive indices. Direct hipBLASLt index is ONE LESS.
Pinned installed rocBLAS source8d1ae90e hipblaslt_host.cpp:446-450/675-678
explicitly subtracts1 before getAlgosFromIndex:
https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocblas/library/src/hipblaslt_host.cpp

Initial run12492 failed before screening because it hardcoded24576 rows. Actual
host plan has24575 valid owned rows,of which24563 require length>512 scoring.
Fixed oracle derives activity from existing c4 lengths and checks unique owner
coverage. Failed source/log preserved in planner-count-failure.tar.gz. Successful
screen session23804 exit0. Do not call this a model or production planner bug.

## Shape/input confirmation

confirm.py session5946 exit0:5 global M(8192,16383,32765,32767,32768),8 owners,
3 q_lora amplitude/order variations,2 algorithms:240/240 raw projection checks
exact to CURRENT full-M torch projection. First two dimensions use actual input
prefixes;32768 appends one real row to32767. These are component shapes,not
newly captured service batches. Exactness masks only genuine length>512 rows.

Paired4-round graph ABBA,full projection vs owner gather+projection,3953:

| Global M | Full ms | Owner ms |
| ---: | ---: | ---: |
|8192|1.206607|0.249951|
|16383|2.363054|0.473528|
|32765|4.678720|0.938976|
|32767|4.673200|0.937963|
|32768|4.671840|0.938043|

Only raw projection(+candidate gather) timed. No weights_proj/RoPE/Hadamard/
quant/cache/logits/TopK/collectives or service timing claim. Extra reduction
traffic/padding can still alter complete-chain benefit.

## Explicit per-stream Lt ABI, no process-global backend flag

direct_lt.cpp provides minimal C ABI over installed hipBLASLt C/C++ APIs:
create descriptors, getAlgosFromIndex(3952), matmulIsAlgoSupported, run on explicit
HIP stream, destroy. Shape is row-major X W^T represented column-major W X^T.
BF16 matrices/output,FP32 accumulate; no weight copies or precision changes.
build.py compiles with /opt/rocm/bin/hipcc,gfx90a,links libhipblaslt. Build-v1
manifest freezes source/output/library hashes,commands and ldd. No installed
ROCm/AIter files modified. Shared object excluded from evidence; clean rebuild
script retained. Library resolves to core-10.0/lib/libhipblaslt.so.1.4; algorithm
ID cannot be assumed portable to another library build.

direct.py owns plans per shape and explicit device/stream; never changes global
BLAS/environment. It shares a max-sized workspace across its plans rather than
per-layer copies. This is still experiment-local,not imported by production.

abi.py first run54014 passed72 ordinary checks then hit its stream guard during
torch.cuda.graph because the harness used the implicit capture stream. Failure
source/report/log preserved in abi-stream-guard-evidence.tar.gz. Wrapper guard
was NOT relaxed. Harness now builds a separate plan owned by the capture stream.

Retry60090 exit0,with ROCBLAS_USE_HIPBLASLT UNSET and global preference unchanged:
-72 checks(3M x3input variations x8owners),all exact to current full projection;
-100 changed-input HIP graph replays,all exact on valid rows;
-independent stream/plan exact;
-cold plan creation precedes any torch GEMM in this process;
-workspace bytes0 for tested localM768/3072,N8192,K1024.

No whole-model or arbitrary-library deterministic claim. No service results yet.

## Next concrete task

1. Promote the hash-checked C ABI/wrapper into a separately default-off fixed-Lt
   producer path. Use backend/runner+thread+stream ownership as the old producer
   does; no global ROCBLAS_USE_HIPBLASLT or torch preference changes. Validate
   actual loaded library/build identity and supported shape before use.
2. Keep original query16 host_plan and full-M weights_proj/full compressor.
   Preserve current owner K32 score path,not old producer's K16-only dispatch.
   Keep initial coveragewidth512..2048/ordinary V4 TP8 EXTEND; long/wider paths
   must fall back BEFORE any collective/query/cache work.
3. Crucial check-mode reference must use ORIGINAL indexer.wq_b/compute_q,NOT
   full-M recomputation with the same candidate handle (old producer check did
   the latter). Compare FP8 Q,folded weights,scores,logical/physical IDs/order
   across all8 ranks and all real C4 layers. Preserve complete cache updates.
4. Only after actual-hit/exact live oracle: identical corrected first-MHC-table
   baseline,real C16x8K,1M pool,noninstrumented ABBA,teacher forcing and long
   outputs. No global default promotion before that. Candidate gain is a
   budget from eliminating replicated query math,NOT a promised5x whole model.

Keep old default-off approximate producer behavior isolated. Add mode/scope
truth-table and no-global-backend-mutation tests. Do not run CPU pytest during
service measurement; earlier CPU imports registered a GPU owner. Confirm GPU
freedom and owned-stop states before each sequential service/component phase.
