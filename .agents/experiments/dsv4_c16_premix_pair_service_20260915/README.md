# Default-off paired-column pre-mix service trial

Status: running; no E2E benefit claimed before the completed ABBA and answer review.
Integration checkpoint: afc913eb48. A1 has started, passed readiness and France.
The first formal control wave reproduced 6959.45 input tok/s; this is one wave,
not the completed A1 median. Sweep session 35562 is executing A1/B/A2 serially.

Original V4, TP8/EP1, native AR, original weights, 1,048,576 logical KV,
32K prefill budget, C16 x approximately 8K fixed public-code inputs (131069 tokens).
All arms use query16/runtime-M, mix8, post-fused4 and empty-tile skipping.
Wide C4, config20 and comb-refine20 are explicitly disabled.
Only `SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS` changes: A1=0, B=1, A2=0.

Fresh process per arm; one warm wave, three formal waves per leg A1/B1/B2/A2,
France check and two separate 128-token answer waves per process. Fixed input IDs,
unique cache salts, zero actual prefix hits. The client verifies exact echoed IDs,
records raw first-token timestamps and uses last-first minus earliest-begin (not drain).
All eight ranks must log candidate selection in B and none in A.
Source hashes are frozen during timing. Process shutdown uses PID plus birth-time ownership.
Existing control output drift means inter-arm completion differences alone cannot be
attributed to this kernel; component exactness and bounded answer review are separate checks.

Prerequisites: six CPU scope tests, three client tests and the production-wrapper GPU
oracle in `../dsv4_c16_premix_pair_20260915/integrated.json` must pass.
No overlapping GPU experiments. Run with the DS conda Python:

```
/home/pc/anaconda3/envs/DS/bin/python .agents/experiments/dsv4_c16_premix_pair_service_20260915/sweep.py
```
