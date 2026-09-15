# Default-off paired-column pre-mix service trial

Status: completed and accepted for narrowly scoped profile promotion.
Integration checkpoint: afc913eb48; source digests match across all arms.
Sweep session35562 exited0; all three owned services stopped with no remaining
processes, and all eight GPUs were confirmed free.

A1/B1/B2/A2 medians:6959.4535 /7368.9612 /7366.3857 /6964.3819 input tok/s.
Mean leg medians: **6961.9177 -> 7367.6734 (+5.8282%)**. All192 formal input echoes
and96 quality echoes matched exact frozen IDs, with zero cache hits. No compile
or exception lines were recorded in the formal intervals. Both candidate quality
waves repeat16/16 and match a control output for every case. Control A1 repeats
15/16; A2 repeats16/16. Read manual-review.md for the bounded-review limitations.

`summary.json` is the timing-closure snapshot before manual review, so its pending
review label is historical; `quality-review.json` and `manual-review.md` close that
gate. These results do not establish global numerical determinism. No new kernel
workspace or weight precision change was introduced. The combined TP8
multi-request/prefill profile now enables pairing by default, leaving explicit0
intact. Post-promotion CPU scope/default suite:46 passed; client tests:3 passed;
bash syntax and diff checks passed. No post-promotion arithmetic change was made.

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
