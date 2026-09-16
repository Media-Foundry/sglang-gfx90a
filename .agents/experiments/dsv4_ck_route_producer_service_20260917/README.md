# TP8 C16x8K route-producer service acceptance

Original DeepSeek-V4-Flash, TP8/EP1/no-A2A, native AR, original checkpoint,
1M logical KV, 32K prefill budget,16 heterogeneous public-code requests,
131069 actual input tokens/wave,zero prefix hits. Only route producer differs;
K32 indexer/common FP32/20 MHC and all prior accepted paths remain fixed.

| Leg | Median input tok/s (3 scored waves) |
|---|---:|
|A1|10270.913070|
|B1|10375.279163|
|B2|10370.529876|
|A2|10280.087707|

Control10275.500388, candidate10372.904520, gain0.947926%.
Control drift0.089326%, largest within-leg range0.432763%. This finite-wave
screen is not a confidence interval. No decode/DSpark throughput claim.

All8 ranks selected route producer only in B;1Mpool retained.192 full128-token
answers agree across processes/waves.1008 teacher-forced positions have exact
logprobs and Top5 records across A1/A2,A1/B and prior accepted K32/B.
Separate live diagnostic compared1376 full routed outputs byte-exact.
Its6667tok/s includes duplicated reference work and is not scored.

No universal batch-invariance claim. Original weights/arithmetic are retained;
only the stage1 output address/route ordering and stage2 consumption change.
The new selector remains default-off, restricted to native ordinary large
prefill M16384..36864,I256 and verified CK contracts. Decode/draft/spec/V4.1
are excluded. Added component peak~265MiB/GCD; real service retained1Mpool.

`validated-launcher.sh` is a byte copy of measured B's launcher. It relies on
the source-hashed independent route-stage1 and unique-Set-stage2 manifests,
not an arbitrary installed CK build. `summary.json` reconstructs raw client
timestamps; `acceptance.json` applies numerical/path/spread gates.
All three services stopped normally; session53179 exited0, all8 GPUs idle.

Next: separate16K/32K regression; their current accepted numbers remain
10151.650740/9854.324387 until new evidence is complete.
