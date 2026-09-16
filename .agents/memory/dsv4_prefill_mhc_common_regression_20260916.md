# Common MHC regression: 8K passed; 16K running

## Latest authoritative progress

8K all three processes stopped and analyzer completed. A1=10225.780574,
B1=10213.640110,B2=10197.732150,A2=10210.579711 input tok/s.
Control10218.180143,candidate**10205.686130**,delta**-0.122272%**,
control drift-0.148652%. No material throughput regression; matches historical
10205.900816. All192 128-token continuations exact per request; both A1/A2 and
A1/B teacher comparisons1008/1008 exact selected-token logprobs and Top5 records.
Historical direct-dequant B versus newA1 also16/16 full128-token answers exact.
summary-8k.json has reconstructed raw timing, scoped paths and numerical records.

16K control A1 PID2607054 confirmed live (new service process, ready), orchestrator
2585923/session88438 still running. First control three waves10023.012013,
10023.857027,10017.121310, median10023.012013. First quality wave complete at
last observation. Do not restart; run.py will finish B1/B2/A2 serially.
Full16K acceptance and complete regression archive still pending.

Prepared non-scoring profile harness in dsv4_prefill_length_profile_20260916,
committed/pushed26c646a132. No GPU profile launched alongside this regression.
CPU analyzer reproduced historical8K analysis exactly and passed synthetic
4/8/16-forward grouping, complete/nested closure and missing-rank rejection.
Next profile should measure latest same configuration per length, not reuse
old13.079s/8K budget as if direct-dequant and common32K policy were absent.

The following notes preserve initial setup; old initial PID/status superseded.

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
