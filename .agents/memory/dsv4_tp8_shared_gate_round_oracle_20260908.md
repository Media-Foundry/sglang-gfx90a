# TP8 C1 shared gate/up + activation exact-rounding oracle

Production selector unchanged, live service3013729 not restarted.
PhysicalGPU4 only; amd-smi found no external GPU processes before testing.
43 independent synthetic BF16 [512,4096] weight matrices (172MiB total temporary)
avoid a single-weight L2-only benchmark; no service/KV allocation displaced.

The old fused gated HIP kernel used unroll1 and omitted BF16 rounding. This
experiment uses unroll2 to match ordinary TP8 GEMV, rounds completed gate/up
dot products to BF16, then rounds SiLU to BF16 before multiplying the BF16 up
value. Clamp limit10, production Torch activation sequence retained.
`kTorchRound=false` is the default template argument, so existing users keep
their prior math; only the uniquely named oracle JIT module instantiates true.
No production shape guard was expanded.

Five ABBA cycles,43-operation graphs,20 replays/sample:

| Rows/waves | ordinary GEMV + Torch activation us | fused us |
|---|---:|---:|
|1/4|13.6738|6.6761|
|1/8|13.7084|7.5913|
|2/4|13.7072|7.7427|
|2/8|13.6838|10.9157|

All four configurations:100/100 activation mutations exact against the current
ordinary GEMV + Torch clamp/SiLU/multiply, weights mutated every25 iterations;
max_abs0. All43 captured output buffers remain exact after1000 graph replays.
These are synthetic fixtures, not full-model or all-real-layer correctness.

Best component reduction~51.2%, saves~7us per layer if exposed;43 layers gives
~0.30ms/token upper budget (~2.5% of an84tok/s step). Shared/routed overlap can
hide most or all of this; no E2E claim. Next test a default-off TP8/M1/native
decode selector with real weights, C1/C32 regressions and fixed-prefix checks.
C32 is not a shape supported by this candidate.

Reproduce:
`HIP_VISIBLE_DEVICES=4 /home/pc/anaconda3/envs/DS/bin/python scripts/rocm/bench_dsv4_tp8_shared_gate_round_oracle.py`
