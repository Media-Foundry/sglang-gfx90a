# TP8 DSpark M128 no-A2A EP2/expert-TP4 component rejection

Goal: keep attention TP8 and total packed routed-weight bytes/GCD constant,
but split experts into two concurrent owner groups (E128/I512) instead of the
current E256/I256 layout. No token A2A was included.

Oracle: `scripts/rocm/bench_dsv4_tp8_dspark_m128_noa2a_ep2.py`, physical GCD4,
non-LDS A4/R2/G832/D832. It includes gate/up, intermediate quant, masked partial
clear, down and fixed reduction. It excludes sorter, initial quant, final TP8
collective and shared-expert interference. Independent synthetic weights mean
this is a cost screen, not cross-layout output equivalence.

Single component ABBA results:

- balanced: 998.526 us baseline; 996.828 us slowest owner; +0.170%
- skewed: 979.053 us baseline; 971.589 us slowest owner; +0.768%
- each shape ran 100 input mutations and 1000 graph replays without replay drift

Decision: reject production wiring. Both results are far below the 10% gate,
and real sorter/ownership integration could only reduce the margin. This also
agrees with the earlier M32 layout screen: halving expert tasks while doubling
the local expert width does not materially reduce the routed critical path.
