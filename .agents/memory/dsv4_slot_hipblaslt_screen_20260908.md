# Explicit hipBLASLt row-invariance screen

Unchanged diagnostic server2878897,1M pool. PhysicalGPU4 only, server idle;
amd-smi PID set contained only its process tree before experiment. No global
AITer tuned-table writes, no new production selector or weight cache.

Using real layer0 normalized736x4096 BF16 activations (16identical46-token
prefixes) and1536x4096 BF16 wqkv_a, aiter.hipb_findallsols returned2226supported
solution IDs. Each was invoked explicitly through hipb_gemm and checked for
finite output and equality across identical rows BEFORE timing.

All2226 failed row invariance; nonfinite0, reference-error rejections0,
stable candidates0. Second independent scan added explicit rejection counters
and reproduced this result. No finalists qualified for mutation/graph ABBA.
Do not claim any of these solutions is faster or numerically validated here.
The test covers this shape/weight/input/library build, not every possible CK
implementation. It establishes that substituting an existing hipBLASLt ID
does not solve this observed contract. Continue with explicit fixed-order
MFMA/CK implementation; previous Triton oracle remains correct by its limited
row-invariance test but26%slower at best. Full E2E repair remains incomplete.

Raw logs /tmp/dsv4_slot_hipblaslt_20260908.log and
/tmp/dsv4_slot_hipblaslt_reasons_20260908.log; script
scripts/rocm/bench_dsv4_slot_hipblaslt.py. No model/kernel production edits.
