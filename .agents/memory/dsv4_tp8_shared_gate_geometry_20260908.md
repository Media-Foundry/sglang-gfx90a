# TP8 shared-expert audit and C1 fixed-unroll screen

Production unchanged. Existing C32 N512 hipBLASLt replacement was already
rejected end-to-end in experimental_switches.md (2026-08-27), including the
shared/compressor aliasing hazard. Do not retry that global tuned row blindly.

C1 DeepseekV2MLP fused gated path only admits N4096/N1024, not TP8 N512.
This is not enough to label a bug: existing gated HIP kernel clamps FP32 dot
results directly, whereas the ordinary projection produces BF16 before the
activation. Its unroll1 also differs from current ordinary GEMV unroll2.
Simply extending the shape guard would silently alter numerical semantics.

Screened ordinary N512/K4096 GEMV with fixed unroll2, no new production selector.
Physical GPU4, amd-smi all GPUs idle.43 independent synthetic4MiB BF16 weights
(172MiB temporary) to avoid one-weight L2-only timing. No service/KV allocation
was displaced. Five ABBA cycles,43-node graph repeated20 times per sample.

|rows/waves|A us|B us|
|---|---|---|
|2/8 self|6.409|6.418|
|1/4|6.402|6.230|
|1/8|6.410|6.101|
|2/4|6.417|6.416|
|1/16|6.416|6.282|
|2/16|6.405|10.101|

All six:100 activation mutations exact,max_abs0; graph final output exact,
stable after1000 replays. Does not test real layer activations or E2E parity.
Best saves0.309us per projection (~13.3us across43 layers if entirely critical,
~0.11% of C1 at84tok/s). Shared work may be hidden by routed experts. No C32
claim: ordinary wave GEMV wrapper only accepts M<=4. Keep production unchanged;
next worthwhile shared fusion must preserve BF16 round-trip and measure the
whole shared/routed join, not just a projection.
