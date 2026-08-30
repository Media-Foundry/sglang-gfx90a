# DSV4 TP4 BS32 delayed-sampling service A/B

Date: 2026-08-30

The current TP4 BS32 profile already enables scheduler overlap, single-batch
overlap, ROCm multistream execution and graph tiers 1--32.  This experiment
tested the remaining default-off `SGLANG_ENABLE_DELAY_SAMPLE` scheduler option,
which moves the sampling closure until after the previous result is processed.

Both independently started services used four GCDs, original weights, native
AR, 32 distinct fixed input-ID requests and five 512-token streaming rounds.
The 32-row next-token record was JSON-identical to the accepted issue-order-3
oracle for both services, including IDs, full logprob rows and top-5 entries.
All requests completed 512 tokens and passed the France first-nine check.

| profile | HTTP resident samples (tok/s) | median | trimmed mean |
|---|---|---:|---:|
| delayed sampling | 627.599 / 626.351 / 627.289 / 627.346 / 627.450 | 627.346 | 627.362 |
| default sampling | 629.177 / 628.492 / 629.481 / 630.109 / 629.230 | 629.230 | 629.296 |

After the first metrics-delta warmup, scheduler decode was about 688 tok/s for
the delayed service and 695--696 tok/s for the default service.  Delaying
sampling therefore regresses the HTTP resident metric by roughly 0.31% and the
scheduler model metric by roughly 1.1%.  Keep the option disabled; it delays a
critical dependency rather than removing a host submission seam.

The concurrent harness was also extended with a same-service
`--stream-interval-sequence` mode and optional scheduler/device metrics.  A
server without `/metrics` now falls back cleanly instead of failing the
benchmark.  This diagnostic separates HTTP/SSE drain from model decode and is
not itself a performance optimization.
