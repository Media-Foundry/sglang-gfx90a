# TP8 strict DSpark M128 gate-grid rejection (2026-09-10)

Scope: original checkpoint weights, TP8/EP1/no-A2A, strict full-target
DSpark gamma=3, C32 real heterogeneous code requests, 1M-token pool.  Only the
row-prefetch routed gate grid changed; down remained at 832 blocks.

## Component screen

Real route histograms showed small isolated layer wins for gate=1040 versus
gate=832 (roughly 0.1--1.1% on learned-router layers).  All component trials
passed 100 input mutations and 1000 HIP graph replays bitwise exactly.

## Service ABBA

Order was A(832), B(1040), B(1040), A(832), with a fresh process and warm run
for each arm.  Measured resident decode throughput:

- A1: 1084.2423 tok/s
- B1: 1064.1724 tok/s
- B2: 894.2014 tok/s
- A2: 1069.1008 tok/s

All 32 real-code requests in every arm passed the severe-repetition gate.  The
B mean was about 9.1% below A and showed a large second-process tail.  The
component gain therefore does not survive the full graph/rank-max execution.

Decision: reject the configurable 1040-block service path and restore the
production fixed G832/D832 geometry.  Keep the stat-recorder input support in
the standalone geometry oracle for future real-distribution analysis.

Artifacts:

- `/tmp/dsv4_tp8_m128_gate1040_abba_retry_20260910/`
- `/tmp/dsv4_tp8_dspark_current_route_20260910/analysis.json`
