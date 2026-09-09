# TP8 DSpark M128 A8 and A4-grid component screens

Physical GCD4, TP8 I256 M128, non-LDS lookup, synthetic balanced and skewed
routes. Every arm passed 100 activation/router-weight mutations and 1000 graph
replays bitwise exactly. Timings use one symmetric multi-arm pass and cover the
full routed chain after the initial input quantization.

## A8

Best balanced A8 (G2080/D1248) improved 1036.492 -> 971.922 us (+6.644%),
but the same shape regressed skewed 994.144 -> 1000.152 us (-0.601%). The
least-bad skewed A8 (G2080/D832) gained only 0.722%. Reject a fixed A8 selector:
the result is distribution-dependent and repeats the prior TP4 warning.

## A4 grid

G832/D1248 was the only candidate positive on both synthetic distributions:
balanced +2.197%, skewed +0.356%. This is too small for another costly service
cycle and below the 5% component gate. Keep production G832/D832.

The screen remains available through
`scripts/rocm/bench_dsv4_tp8_dspark_m128_geometry.py --screen-a8/--screen-grid`.
