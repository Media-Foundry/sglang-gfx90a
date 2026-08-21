# DSV4 gfx90a 实验开关清单

范围：本地 bring-up/优化提交 `505b337379`、`2a98bfbb1f` 以及
`scripts/rocm_dsv4_flash.sh`。这里不把目标仓库的普通上游开关当成我们新增的
调试开关。当前测试口径仍是 TP4/EP4、Mori、batch=1、原生 AR。

## 优先关注

| 开关 | 当前 harness 默认 | 作用 | 风险/备注 |
| --- | ---: | --- | --- |
| `SGLANG_DSV4_GFX90A_MORI_SHARED_EXPERT_TP` | `1` | 将 shared expert 按 TP4 分片，和 routed MoE 重叠，最后做一次 shared partial + routed gather | 最高风险；涉及 side stream/event 与 graph 内 all-reduce，是此前 capture 自旋的主要边界 |
| `SGLANG_OPT_USE_TRITON_MHC_COMBINE` | `1` | 使用 gfx90a Triton MHC weighted-sum/post-combine | 中风险；应和 Mori/无 A2A 分开 A/B |
| `SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX` | `1` | gfx90a MHC pre-mix Triton | Mori 下由 `_A2A` 和 `_MAX_BS` 分 tier 控制；bs1/2/4 已验证，bs8 必须回退 AIter 才能稳定 capture |
| `SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A` | `1` | 允许 Mori decode 使用 Triton pre-mix | 必须同时保留 `MAX_BS=4`；无限制会让 tier-8 graph 四 rank 同步自旋 |
| `SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_MAX_BS` | `4` | Triton pre-mix 的最大全局 graph tier | `0` 表示无限制；当前生产候选为 4，tier8 使用 AIter MHC |
| `SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR` | `1` | 将 DSV4 attention projection 的 block-FP8 权重缓存为 BF16，匹配 gfx90a grouped GEMV | 约增加 1 GiB/GPU；kernel 只覆盖固定 decode shape，失败应回退 einsum |
| `SGLANG_DSV4_GFX90A_BF16_SHARED_GATE_UP` / `_DOWN` | `1` / `1` | shared expert gate/up、down 权重 BF16 cache | 约增加显存；必须在权重加载前设置 |
| `SGLANG_DSV4_GFX90A_FUSED_SHARED_GATE_UP` | `1` | 将单 token gate/up 与 bounded SwiGLU 融合 | 依赖 AIter gated GEMM；需要单独做数值/graph A/B |
| `SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS` | `8` | 将全局 bs1 的 native Sinkhorn 从 20 次减到 8 次 | 仅全局 bs1 生效；bs2/4/8 固定 20，避免改变多请求 expert balance |
| `SGLANG_DSV4_GFX90A_FUSED_MHC_WEIGHTED_RMS` | `1` | 融合 bs1 weighted residual sum 与 RMSNorm | BF16 输出与分离路径 bitwise exact；多 token 保留原并行路径 |
| `SGLANG_DSV4_GFX90A_SPLITK_MHC_PRE_MIX` | `1` | bs1 MHC 24x16384 FP32 pre-mix 使用 8-way split-K | 48 CTA 配置给 Mori progress kernel 保留 CU；不要改回 192 CTA scalar-row graph |

## 其他本地开关

- `SGLANG_DSV4_GFX90A_BF16_SHARED_EXPERT`：同时视 shared expert 为 BF16 cache
  的总开关；当前 harness 设为 `0`，但 gate/up/down 三个细开关仍为 `1`，语义容易
  混淆，后续应统一成一种配置表达。
- `SGLANG_DSV4_MORI_ROTATE_SHARED_EXPERT_OWNER`：按 layer 轮换物理 rank 到逻辑
  token chunk 的 owner；shared-TP 开启时 harness 默认 `0`，TP1 fallback 时默认 `1`。
  不增加 collective，但会改变每层 owner 映射。
- `SGLANG_DSV4_USE_BF16_RMSNORM_WEIGHT`：将 DSV4 RMSNorm 参数改存 BF16；默认关闭，
  未纳入主 harness，属于权重 dtype/数值实验。
- `SGLANG_OPT_USE_TRITON_INDEXER_POST`、`SGLANG_OPT_USE_TRITON_INDEXER_FULL`：
  DSV4 indexer Triton post/full 路径；harness 默认开启，属于性能路径而非单纯 debug。
- `SGLANG_OPT_USE_AITER_MHC_PRE`、`SGLANG_OPT_USE_AITER_MHC_POST`：harness 默认关闭，
  用于 AIter MHC 路径 A/B。
- `SGLANG_DSV4_MHC_PREWARM`、`SGLANG_OPT_USE_TRITON_FUSED_MHC`：当前源码中只有
  `environ.py` 定义，没有实际读取点，疑似遗留/死开关，暂不应作为实验结论依据。
- `SGLANG_OPT_USE_AITER_INDEXER`：当前 `environ.py` 有重复定义（通用 DSV4 Aiter 区域
  和 DeepSeek V4 区域各一处）；后续应去重，避免后定义覆盖前定义造成误解。

## Mori / graph 调试开关

- `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`：harness 默认 `256`，代码通用默认
  `4096`；改变 capacity/geometry，不能和速度结果混口径。
- `SGLANG_MORI_DISPATCH_DTYPE`、`SGLANG_MORI_COMBINE_DTYPE`：harness 默认 BF16，
  还支持代码中的 FP8/FP4/auto 变体。
- `SGLANG_MORI_USE_EXTERNAL_INP_BUF`：harness 默认 `0`，代码默认读取为 `true`；
  这是一个明确的默认值反差，应在脚本注释和代码语义中统一。
- `SGLANG_MORI_INTRANODE_COMBINE_BLOCK_NUM=32`、
  `SGLANG_MORI_INTRANODE_COMBINE_WARP_NUM_PER_BLOCK=4`：只调 combine geometry；
  dispatch geometry 另由 `SGLANG_MORI_INTRANODE_BLOCK_NUM` 和
  `SGLANG_MORI_INTRANODE_WARP_NUM_PER_BLOCK` 控制。
- `MORI_ENABLE_SDMA`、`MORI_DISABLE_TOPO`、`MORI_DISABLE_AUTO_XGMI`、
  `MORI_SHMEM_HEAP_SIZE`、`DEEPEP_MODE`：通信/拓扑运行时开关；`MORI_ENABLE_SDMA=1`
  还会设置 `LD_PRELOAD`，必须独立验证，不能与普通 Mori 结果混合。
- `DISABLE_CUSTOM_ALL_REDUCE`、`DISABLE_ATTN_TP_GATHER`：通信路径 A/B；前者影响
  shared partial/attention 的 collective，后者影响 decode graph 的 padded gather。
- `ENABLE_SINGLE_BATCH_OVERLAP`、`DISABLE_DECODE_CUDA_GRAPH`、
  `ENABLE_PROFILE_CUDA_GRAPH`：实验/诊断开关，不改变“单请求”定义，但会改变 graph
  或 profiler 口径。

## AR 口径保护

`rocm_dsv4_flash.sh` 会拒绝非零的 `SPECULATIVE_*` 环境变量，包括 DSpark 参数；
因此任何 benchmark 结果都不应把 accepted-token 或 verify throughput 混入原生 AR。

## 2026-08-21 graph-8 checkpoint

- graph tiers 为 `1/2/4/8`，`max_total_tokens=8192`，`swa_full_tokens_ratio=0.65`。
  `mem_fraction_static` 必须保持 `0.80`；降到 `0.76` 会让可用 token bytes 少约
  2.56 GiB，full pool 从 8192 被压到 2304，最多只能准入四条 256-token 请求。
- `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK=16`，
  `AITER_GFX90A_MXFP4_QUANT_MAX_ROWS=64` 覆盖 graph-8。
- gfx90a 的 `mhc_fused_post_pre` 现在对大 token prefill 也使用 Triton/native
  decomposition；旧的 TileLang MFMA 路径在 256-token extend 上无法 lowering。
- 同一 hybrid 服务实测单请求热态 `50.27 tok/s`，8 并发 `190.46 / 186.95 tok/s`；
  全 AIter MHC 的 8 并发上限为 `220.42 / 200.77 tok/s`，但单请求约 35 tok/s。

## 2026-08-21 bs1 MHC split-K checkpoint

- native Sinkhorn 8 次相对 20 次的 comb 最大误差约 `6.1e-5`；weighted-sum +
  RMSNorm 融合为 bitwise exact；split-K pre-mix 相对单段归约最大误差约 `1e-6`。
- split-K 使用 8 个 K 分片、4 rows/CTA，共 48 CTA。192 CTA 版本的 standalone
  microbenchmark 更快，但会让 Mori graph capture 的设备端通信 progress 失步。
- 全局 batch size 必须作为 8/20 Sinkhorn 的 rank-invariant 判据；使用 rank-local
  token shape 会让四 rank 在 graph capture 内自旋。
- 原生 AR 复验：单请求热态 `50.72 / 52.32 tok/s`；8 并发
  `219.78 / 207.92 tok/s`。所有请求均输出 256 token 且 `finish=length`。

## 建议的下一步 A/B 顺序

1. 固定 `MORI=normal`、capacity=256、custom all-reduce 和 graph 配置，只切
   `MORI_SHARED_EXPERT_TP`。
2. 在 shared-TP 稳定后，分别切 `BF16_ATTN_LINEAR` 和 shared gate/up/down/fused，
   每次只改一个开关。
3. `MHC_PRE_MIX` 仅在 no-A2A 单独测试；不要用它解释 Mori 结果。
4. 清理死开关、重复 `SGLANG_OPT_USE_AITER_INDEXER` 和 external-input-buffer 的
   默认值/注释后，再将脚本作为可复现实验入口。

## 2026-08-21 FP16 MHC + inverse-RoPE checkpoint

- `SGLANG_DSV4_GFX90A_FP16_MHC_DOT=1` 只把 bs1 MHC split-K dot 的缓存权重降为
  FP16，仍以 FP32 累加；相对 FP32 权重的 microbenchmark 最大绝对误差约
  `1.4e-3`。BF16 权重误差约 `1.4e-2`，因此没有采用。
- `SGLANG_DSV4_GFX90A_FUSED_MHC_SPLITK_TAIL=1` 将 split-K reduce、8-iteration
  Sinkhorn、weighted residual 和 RMSNorm 合并为一个 gfx90a Triton tail。
- `SGLANG_DSV4_GFX90A_FUSE_ATTN_INVERSE_ROPE=1` 将 DSV4 decode 的 inverse RoPE
  放入 unified-KV attention epilogue；`kv_splits=1` 与 split-K、T=1 与 T=8
  均已和原独立 RoPE 路径逐元素 bitwise 对齐。
- 同一服务的 256-token 原生 AR probe：干净基线单请求稳态中位数约
  `49.65 tok/s`，组合候选稳态为 `53.50 / 53.81 / 54.13 / 54.04 / 54.14
  tok/s`，中位数 `54.04 tok/s`（约 `+8.8%`）。
- 8 并发回归为 `218.45 / 217.96 tok/s`，保持高于 `180 tok/s` 验收线。
