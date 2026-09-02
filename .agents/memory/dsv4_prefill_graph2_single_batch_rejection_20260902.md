# DSV4 C32 graph-tier-2 single-batch attempt (2026-09-02)

## Hypothesis

The earlier one-forward C32 BF16-CK attempt missed allocation by about 432 MiB.
Reducing decode graph capture from BS8 to BS2 might release enough memory for a
73,724-token prefill while preserving graph decode at C1/C2.

## Configuration

- original checkpoint, TP4/EP1/no-A2A on physical GCDs 4--7;
- prefill throughput profile enabled;
- `CUDA_GRAPH_MAX_BS_DECODE=2`;
- chunk/max-prefill tokens 73,728 and prefill request cap 32;
- 32 distinct code-review requests from the audited manifest;
- one generated token for the prefill wall-time probe.

France returned `Paris.` before the C32 run.

## Result

The service started, but did not form one stable 73,728-token forward.  Request
arrival and SWA/KV admission produced irregular large shapes such as M=43,773
and M=46,077, followed by small tails.  Each previously unseen exact M also
entered untuned BF16 GEMM shape setup.  Two attempted C32 rounds took about
124.65 and 132.99 seconds, only 591 and 554 aggregate input tok/s.  All 32
requests in the first completed round returned one token; the second client run
was stopped after its first completed round because the rejection was decisive.

Reducing graph tiers therefore releases memory but destroys the stable
12+12+8/M27648 service geometry and is rejected.  Keep the accepted BS8 graph
capture and request-12 admission profile.  A future one-forward design must
provide deterministic admission and stable bucketed large-M kernels rather
than exposing arbitrary exact M shapes to runtime GEMM selection.
