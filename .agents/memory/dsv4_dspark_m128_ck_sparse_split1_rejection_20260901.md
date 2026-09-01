# DSpark M128 CK sparse-decode split=1 rejection (2026-09-01)

The accepted gfx90a CK/MFMA sparse-decode kernel uses split=2. Because M128
already provides 128 token CTAs, an oracle-only split=1 entry tested whether
halving FP32 workspace traffic outweighed the loss of KV-scan parallelism.

Physical GCD 4, M128/H16/D512, 100 random input mutations and 1000 graph
replays passed the existing attention tolerance and graph-stability gates.

| visible KV rows | split=2 | split=1 | split=1 change |
|---:|---:|---:|---:|
| 128 | 76.637 us | 89.274 us | -14.2% |
| 256 | 118.928 us | 143.155 us | -16.9% |
| 512 | 181.286 us | 241.010 us | -24.8% |

Split=1 loses increasingly as context grows. The longer serial KV scan is more
expensive than the second split's workspace and reduction. The temporary C++,
Python, and benchmark entry points were removed; retain split=2 in production.
