# Common MHC shorter-length regression: running, not yet accepted

2026-09-16.32K common FP32/20 service acceptance committed/pushed34fa7b0e9c:
8667.433540 ->9605.065395 input tok/s (+10.817872%),192 continuations exact,
1008 teacher-forced positions exact vs new same-contract reference. Legacy
FP16/8 differences retained separately; no universal invariance/default claim.

Frozen regression harness committed/pushed5f2223df82. Directory:
`.agents/experiments/dsv4_prefill_mhc_common_regression_20260916/`.
Botharms premix-owner1, exactpost/H16/uniqueSet/vec4/directdequant/wide-indexer1,
queryproducer0/MFMA0, nativeTP8/EP1/noA2A, originalweights,1Mpool/32Kchunk.
Only common MHC policy off/on.8K and16K separately each A1/B1/B2/A2 with
three scored waves per leg,4x16x128quality/process,1008fixedteacherpositions.
8K fixture hash checked against committed direct-dequant archive manifest;
16K against official-chat input validation. Quality prompts frozen, cache salted.

Current unified exec session88438; orchestrator2585923,8k-A1 server2585956
confirmed live in weight loading (19:00local), not yet ready/scored. Exact
driver/log names8k-A1-driver.log and8k-A1/P8k-common-regression-A1.service.log.
The orchestrator owns serial cleanup and automatically runs remaining arms and
16K afterward. Do not restart because observation expires. Confirm process
identity/liveness before polling; no concurrentGPU experiments or runtime edits.

No new speed claim until raw client waves and exactness gates are read. If
different input lengths are compared, historical8K10205.90 /16K10024.77 /
32K9605.07 are separate workloads, not a same-length slowdown. Next after this
regression: latest same-rank-envelope profile at relevant lengths, to quantify
where longer L increases cost rather than assuming all difference is attention.
Small-prefill/decode precision contracts remain out of this large-prefill fix.

First script-generation attempt failed in orchestration JavaScript with
TypeError s.index is not a function before any patch. Repeated with indexOf,
applied files, py_compile/diff-check passed. No failed GPU test was hidden.
User/runtime cuda_graph_runner_memory_usage.pickle and unrelated files preserved.
