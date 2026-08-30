# TP4/M32 CK sparse decode experiment (2026-08-31)

The existing TP4/M64 CK-style MFMA unified sparse-decode candidate was
temporarily generalized to M32 and tested on physical GCD 4.  The production
selector was reverted because the gain was too small and its numerical path
was slightly less accurate than the established Triton implementation.

## Standalone ABBA result

Inputs were BF16, `T=32`, `H=16`, `D=512`, with 20 independently mutated
queries per context.  Timings used seven ABBA rounds and 100 graph replays per
sample.

| Context | Triton (us) | CK (us) | CK change |
|---:|---:|---:|---:|
| 128 | 41.067 | 41.109 | -0.10% |
| 256 | 69.366 | 68.251 | +1.63% |
| 512 | 117.804 | 116.016 | +1.54% |

Against the existing Triton output, the maximum absolute difference was
`0.00390625` and the worst relative L2 difference was about `0.00383`.
Against a direct FP32 PyTorch attention oracle at context 128, Triton had about
`0.00201` relative L2 error and CK about `0.00337`.

## Decision

Do not enable the M32 CK path.  The 0--1.6% standalone gain is far below the
5% end-to-end checkpoint threshold, disappears at context 128, and comes with
a somewhat larger reduction-order error.  Retain the M64-only selector and
focus M32 work on larger attention/MoE boundary costs.

The diagnostic harness remains at
`scripts/rocm/bench_dsv4_tp4_m32_ck_sparse_decode.py` for future kernel
iterations.  Correctness must be measured eagerly; retaining several
independent ROCm graph-private allocation pools across context sizes produced
a stale-output artifact in an earlier version of the harness.
