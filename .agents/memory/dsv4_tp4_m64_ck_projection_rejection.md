# TP4 M64 CK attention-projection rejection

Date: 2026-08-30

Four standalone Composable Kernel profiler winners were replayed against the
current hot PyTorch/hipBLAS path with real layer-20 tensors, seven-round ABBA,
32 nodes per timing graph, 100 input mutations, and 1000 graph replays.

| projection | shape N x K | production | CK | CK regression |
|---|---:|---:|---:|---:|
| wqkv_a | 1536 x 4096 | 36.03 us | 120.27 us | 3.34x |
| core compressor | 2048 x 4096 | 38.05 us | 130.04 us | 3.42x |
| index compressor | 512 x 4096 | 36.77 us | 127.32 us | 3.46x |
| index weights | 64 x 4096 | 11.33 us | 136.84 us | 12.07x |

All graph replays were bitwise stable. Mutation worst relative L2 ranged from
about `4.5e-5` to `1.21e-4`, with minimum cosine at least `0.99999982`.
Correctness is therefore not the rejection reason; the CK profiler ranking
does not transfer to the production hot-graph launch/layout context.

Decision: do not connect any CK projection selector. Keep the existing
PyTorch/hipBLAS path and require a real graph ABBA before trusting future CK
profiler winners.

Artifact: `/tmp/dsv4_tp4_m64_ck_projections_abba.json`.

