# Pending bounded route-reducer screen

Prepared while16K/32K service ABBA runs. No GPU compile or timings yet.
Do not run alongside any service. It will refuse an occupied selected GCD.

The accepted route producer writes FP32 partials in route order; the reducer
gathers six route IDs for each token. Compare only three alternatives:
vec4 with wave-uniform address hints,vec8,and vec8 with the same hints.
Fixed1664x256 launch and strict slot0→5 FP32 adds followed by one BF16 cast
are retained. No approximate output,atomic reduction,extra scratch or new
communication. Wider vectors may lose on registers; hints may add latency.

This is NOT the already rejected N-stripe strategy. It does not reduce the
logical partial read/write byte count. Prior vec4 screen only tested token-major
partials; this screen uses the actual route-major producer and inverse mapping.

`test_mapping.py` checks the wave-uniform addressing precondition on boundaries.
`screen.py` will hash-check two historical real MoE fixtures and independent CK
builds,compare full BF16 and FP32 reducer outputs under input/weight/route changes,
test100 graph replays with partial mutation,then graph ABBA both isolated
reducer and complete metadata→stage1→stage2→reduce chains. Sort,dequant and
allocation are excluded from both arms. No service-speed claim from that test.

Only consider a complete-chain winner for a later full-routed helper oracle.
If all three lose, close the experiment without changing production.
