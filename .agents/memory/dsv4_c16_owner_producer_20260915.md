# Owner-query producer: live-object oracle and row-dependent GEMM drift

Accepted service performance remains **7953.956103 input tok/s**, original
V4 / TP8 / EP1 / no-A2A / native AR, original checkpoint, 32K chunk and 1M
logical KV pool. This is progress on the active drift/prefill goal, not a new
accepted speed checkpoint. Production query production remains full-M.

The review's runtime-M, query16, MHC reuse8 and post-optimization profiling
tasks already have separate completed records. Current profile query producer
budget is 0.6273s/wave, while owner scoring/Top-K/exchange/reconstruction is
0.3017s. Do not reuse the old full-local logits budget or multiply a producer
micro speedup into full-service throughput.

## Actual live-object producer oracle

Default-off `SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR` hook is inside the existing
strict original-V4 TP8 ordinary-EXTEND owner guard. It probes first layer20;
all ranks synchronize and rank0 uses the actual loaded C4Indexer while other
ranks wait. Candidate results are discarded. Full production compressor,
cache updates, Q, scores and selected IDs continue unchanged after the probe.
This explicit stop-the-world diagnostic is NOT an E2E timing run.

Fresh process completed 16 exact input echoes, zero cache hits, one completion
per request, then a France/Paris check and owned shutdown. Full first-forward
M32767, extend lengths8192/8192/8191/8192, prefix0. Each of eight owner plans
has3072 valid rows. Entire real x/q_lora/positions/input IDs saved locally;
all actual projection parameter shapes, dtypes and SHA256 recorded.

Full-M producer recomputation matches original Q and weights byte-for-byte.
Three ABBA cycles, five complete producer calls/event: full producer median
**7.40889ms**, owner producer including gathers **0.80101ms**. This is isolated
rank0 live-object timing, not an eight-rank service speedup.

Each compact producer repeats byte-exactly. Compared to full-M selected rows,
however, raw BF16 projection differs by up to0.0078125, leading to FP8 Q,
folded weights, scores and Top-K differences. Owner0..7 positional logical-ID
mismatch counts:392/1189/1037/522/395/1389/925/1203. These are POSITIONAL
mismatches, not membership symmetric differences; one insertion can shift
many ordered IDs. Full ID tensors were not saved by this initial probe, so
membership counts cannot honestly be reconstructed from this report.

FP8 Q code differences alone are not effective-query errors: its quantization
scale is folded into weights. Score max_abs across owners ranges0.02279..0.05958.
Do not describe this as a stale-buffer race or as repaired determinism.

## Frozen real-input projection replay, physical GPU4 only

`projection_replay.py` reads the actual inputs, verifies their manifest hash,
and reconstructs layer20 indexer wq_b from the original safetensors shard.
Crucially its BF16 weight SHA256 matches the live object's exact hash:
`92a8c482b05c8e12da85c50280d63bb3fb21b5890717cb23ad1cc2e372770768`.
This avoids guessing that an offline reconstructed projection matches runtime.

Initial v1 failed before GPU work due to importing the owner helper outside
its package (relative-import error). Fixed by normal package import; preserved
the empty failed output directory rather than claiming a measurement.

Default backend replay (`projection-replay-v2`) recorded actual GPU kernel names:

- Full M32767: hipBLASLt `MT128x256x32`, MFMA32.
- Compact M3072: hipBLASLt `MT128x96x64`, MFMA32.
- Compact padded4096: same kernel name as full-M, still not exact.

Owner0 raw output comparison covers3072x8192 BF16 elements. Compact differs
in1419 elements/1015 rows, max_abs0.00390625. Full-M row reversal (undoing the
permutation before comparison) differs in2182 elements/1576 rows, despite
same M and same recorded GPU kernel name. Compact padded4096 differs in1616
elements/1161 rows, max_abs0.0078125. All fixed-input repeat checks are exact.
This establishes row/order dependence beyond simply switching M's kernel.
It does not identify the low-level accumulation schedule or prove its cause.

## Narrow alternative backend result, NOT production integration

PyTorch's `preferred_blas_library('cublas')` selects ROCm rocBLAS in this build.
In a separate isolated process (`projection-replay-rocblas-v2`), all five
variants use the same recorded rocBLAS `MT128x96x64` kernel. Full/compact,
full/compact reversed and compact padded4096 agree exactly on all selected
elements. Repeats and compact row-permutation check are exact.

But all these rocBLAS results differ from the current default full-M result
in **2023 elements/1476 rows**, max_abs0.00390625. Thus this is an encouraging
ROW-STABLE ALTERNATIVE on one real layer, NOT byte-equivalence to production,
NOT a whole-indexer/whole-model numerical fix.

Serial diagnostic medians: full4.32518ms, compact0.44519ms, padded4096
0.56026ms. These are NOT performance acceptance ABBA; the earlier live-object
ABBA covers the complete producer and is a different benchmark. Actual traces
and source hash are included. ROCTracer emitted duplicate-flow warnings;
do not derive precise timeline attribution from those flow links. GPU event
names and ordinary CUDA-event diagnostic timings are recorded separately.

Next: verify the complete producer (weights projection, RoPE/Hadamard/quant,
scores and membership/order), all owner plans and real layer inputs with the
row-stable candidate. Use an explicit per-operation backend if integrated;
do NOT switch global BLAS preference around production calls or perturb AR,
draft, other streams or other model projections. Then teacher-forced/logit and
fresh E2E quality tests before any rollout. Existing CK atomic-stage2 drift
is a separate already-reproduced issue, not fixed by this query experiment.

## Evidence and validation

Directory `.agents/experiments/dsv4_c16_owner_producer_20260915/`:
`capture.py`, `projection_replay.py`, `capture/oracle/report.json`, capture
manifest/lifecycle logs, projection traces/reports. The336MB real input tensor
stays local with recorded SHA256; small evidence archive excludes it.

Seven CPU tests passed (two default-off/hook-scope tests and five existing
owner tests); one existing pytest asyncio_mode warning. Actual live diagnostic
already exercised the GPU path. No live service remains after the diagnostic.
No launcher defaults or production arithmetic were changed.

Final default-backend replay with the same final script independently reproduced
all element/row mismatch counts above; fixed-input repeats stayed exact. Final
script hash is recorded in both `projection-replay-default-final/report.json`
and `projection-replay-rocblas-v2/report.json`. Python syntax and diff checks
passed. After all isolated probes, amd-smi reports no running processes on
all eight GCDs.

Evidence archive `owner-producer-evidence.tar.gz`:780760 bytes, SHA256
`669641da4eabf468e05b034facc4e2c09dbdb65df0b955fd43e1cf819e38ea24`.
Includes original service request/response identity and lifecycle evidence,
live-object oracle report, scripts, default and rocBLAS projection traces.
