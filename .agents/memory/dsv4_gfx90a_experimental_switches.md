# DSV4 gfx90a 实验开关清单

范围：本地 bring-up/优化提交 `505b337379`、`2a98bfbb1f` 以及
`scripts/rocm_dsv4_flash.sh`。这里不把目标仓库的普通上游开关当成我们新增的
调试开关。当前测试口径仍是 TP4/EP4、Mori、batch=1、原生 AR。

## DSpark gamma-three M128 anchor-only checkpoint（2026-08-31）

- TP4 BS32 `start-dspark` 现在默认 gamma=3，并启用严格 M128 anchor-only
  routed selector。每请求布局为 `[anchor,draft,draft,draft]`；anchor 保留完整
  routed MoE，draft 保留 shared expert 与其他全部模型路径，但省略 routed MoE。
- 命中条件固定为 gfx90a + `TARGET_VERIFY` + BS32 + width4 + M128 + 显式环境
  开关。强制打开开关的原生 AR 服务通过 France 与 32 条异质 64-token 请求，且
  无 speculative 字段，证明 AR 路径不可达。
- 真实异质 32 请求 B-A-B 中位数为 B1 `877.649`、A gamma-one `826.172`、
  B2 `874.359 tok/s`；候选中心 `876.004`，提升 `+6.03%`。无手工覆盖的最终
  默认配置三轮为 `839.533/899.225/887.837`，中位 `887.837 tok/s`。所有轮次
  France 首九 token 精确、语义 Paris，且全部请求生成 256 token、`finish=length`。
- 详细记录：
  `.agents/memory/dsv4_dspark_gamma3_m128_anchor_only_checkpoint_20260831.md`。

### DSpark M128 anchor 物理压缩否证（2026-08-31）

- 服务 B-A-B 曾因 acceptance 波动呈现 B1 `900.911`、sentinel A `832.898`、B2
  `901.845 tok/s`，但 host step 没有稳定缩短，不能归因于 kernel。
- 后续同一 layer20 marker 给出决定性反证：sentinel-only M128 routed FP4 为
  `842--856 us`，物理压成 M32 后完整分支反而为 `935--939 us`。strided gather、
  TopK materialize、额外 quant/sort/output scatter 与实际 M32 runner 组合没有复现
  standalone 的约439us结果。
- 实现、环境变量和默认值已全部撤销；一轮 `1014.922 tok/s` 是 acceptance 高点，
  不得作为性能 checkpoint。审计保留在
  `.agents/memory/dsv4_dspark_m128_anchor_compaction_checkpoint_20260831.md`。

## GPU 实验前置检查（强制）

每次启动性能 probe、服务 A/B 或 profiler 之前，先运行：

```bash
amd-smi process --general --sort-by-pid -g 0 1 2 3 4 5 6 7
```

必须核对可见 GPU 上的 PID、进程名和 VRAM 占用都符合本轮实验预期。尤其检查：

- 上一次服务的孤儿 scheduler / launch_server；
- 因工具超时仍在运行的 standalone benchmark、Triton/CK probe；
- hipcc/JIT 编译进程与缓存锁；
- 四个服务 rank 的显存是否对称。

发现非预期进程时，先按明确 PID 停止本轮产生的残留进程，等待 `amd-smi process`
确认显存释放，再建立干净基线。资源检查未通过时产生的吞吐数据一律作废，不能
用于判断优化、提交或回退。

## 优先关注

| 开关 | 当前 harness 默认 | 作用 | 风险/备注 |
| --- | ---: | --- | --- |
| `SGLANG_DSV4_GFX90A_MORI_SHARED_EXPERT_TP` | `1` | 将 shared expert 按 TP4 分片，和 routed MoE 重叠，最后做一次 shared partial + routed gather | 最高风险；涉及 side stream/event 与 graph 内 all-reduce，是此前 capture 自旋的主要边界 |
| `SGLANG_OPT_USE_TRITON_MHC_COMBINE` | `1` | 使用 gfx90a Triton MHC weighted-sum/post-combine | 中风险；应和 Mori/无 A2A 分开 A/B |
| `SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX` | `1` | gfx90a MHC pre-mix Triton | Mori 下由 `_A2A` 和 `_MAX_BS` 分 tier 控制；bs1/2/4 已验证，bs8 必须回退 AIter 才能稳定 capture |
| `SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_A2A` | `1` | 允许 Mori decode 使用 Triton pre-mix | 必须同时保留 `MAX_BS=4`；无限制会让 tier-8 graph 四 rank 同步自旋 |
| `SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX_MAX_BS` | `4` | Triton pre-mix 的最大全局 graph tier | `0` 表示无限制；当前生产候选为 4，tier8 使用 AIter MHC |
| `SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR` | `1` | 将 DSV4 attention projection 的 block-FP8 权重缓存为 BF16，匹配 gfx90a grouped GEMV | 约增加 1 GiB/GPU；kernel 只覆盖固定 decode shape，失败应回退 einsum |
| `SGLANG_DSV4_GFX90A_INT8_WEIGHT_GEMV` | `0` | 为三个 M=1 投影 shape 追加 per-row INT8 权重缓存并使用 CDNA2 dot4 | kernel-only 分别约快 53%/28%/20%；会增加显存且改变投影数值，必须做完整 AR 正确性和吞吐 A/B |
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
- `DISABLE_OVERLAP_SCHEDULE`：harness 默认 `0`，启用 scheduler overlap；设为 `1`
  时恢复旧的 `--disable-overlap-schedule` 基线。
- `ENABLE_SINGLE_BATCH_OVERLAP`：harness 默认 `1`。必须和 scheduler overlap 配套；
  仅开启 scheduler overlap 会在约 58 tok/s 的快态和 50--52 tok/s 的慢态之间抖动，
  配套后单请求快态可稳定复现。

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

## 2026-08-21 scheduler overlap checkpoint

- 正常 Mori、TP4/EP4、graph tiers `1/2/4/8`、纯 AR 下，同时启用 scheduler
  overlap 与 single-batch overlap。
- 单请求 256-token 两组复验：第一组稳态集中在 `58.18--59.05 tok/s`，第二组
  中位约 `58.48 tok/s`；相对同机干净旧基线约 `54.5 tok/s`，提升约 `7--8%`。
- 8 个独立并发请求 aggregate AR 为 `231.43 / 233.27 tok/s`；所有请求均实际
  输出 256 token 且 `finish=length`。

### Mori decode capacity 分层（2026-08-21，未采用）

- `SGLANG_MORI_DECODE_TIERED_CAPACITY=1` 会按进入 dispatcher 的每-rank token
  行数选择更小的 decode plan；当前 shared-expert TP4 + graph bs<=8 下，所有 tier
  实际最多只有 2 行，因此都安全复用了 capacity=2，而不是原来的 capacity=16。
- graph 1/2/4/8 全部捕获成功，纯 AR 256-token 七轮为
  `57.30 / 58.70 / 58.70 / 58.75 / 50.58 / 56.41 / 58.56 tok/s`。
- 稳态没有超过约 59 tok/s 基线，说明缩小 plan/buffer 没有减少固定 32-block Mori
  progress kernel 的关键开销。该开关保持默认关闭，不作为性能提交。

### Graph upload / Mori CU footprint / FP4 dispatch（2026-08-21）

- 显式 `hipGraphUpload` 可完成所有 graph capture，但七轮只有
  `58.50 / 59.04 / 59.07 / 52.40 / 54.88 / 55.51 / 45.08 tok/s`；它不降低
  每次大 graph replay 的 host traversal，保持默认关闭。
- Mori dispatch+combine 同时从 32 blocks 降至 24 blocks，前五轮为
  `59.32 / 59.66 / 59.76 / 59.77 / 59.68 tok/s`，约 1--2% 正向但不足提交；
  16 blocks 已越过通信/计算让路拐点，结果 `50.52--59.55 tok/s` 且抖动更大。
- gfx90a FP4 dispatch 直通已验证可运行：Triton MXFP4 quant、Mori packed FP4
  transport、AIter `GU_ITLV=0` separated gate/up，且 HSACO 只额外实例化 FP4
  dispatch symbol。七轮为
  `54.53 / 57.09 / 57.27 / 57.29 / 57.24 / 51.17 / 57.27 tok/s`；省掉一次
  stage-1 quant 和减半 payload 仍抵不过 separated gate/up 回退，不能作为默认。
- scheduler overlap 单独开启会出现 50--52 tok/s 慢态，因此两个开关作为一组
  生产默认，不用单次 58+ 峰值冒充稳定结果。

### 下一轮结构优化优先级（2026-08-21）

- full-Triton indexer selector 已在 `c093503759` 接通，但长上下文仍会落地完整
  `[batch, max_seq_len]` FP32 logits，再由独立 Top-K 重读；真正目标是 page/CTA
  local Top-512、二级 merge 并直接输出 physical slot。短 prompt 不能验证这一点。
- 短上下文先做 `SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT=0/2/4/8` 的完整
  scheduler-overlap + graph A/B。必须同时看 Mori progress 和最慢 rank，不能用
  standalone expert GEMM 代替端到端结论。
- KSPLIT 无收益后，转向 `M<=4/8/16` 的 gfx90a wave64 FP4-weight/BF16-activation
  expert kernel，减少 block_m=32 padding、sort/permute 和在线 MXFP4 quant 开销。
- Mori 按 `normal+CU`、`low_latency`、`low_latency+SDMA` 顺序做单变量实验；记录
  dispatch/expert/combine 的四 rank 最慢值。外部 input buffer 直写协议暂不改。
- unified-KV 已消除 64-head attention 算术，但仍可能创建并清零 64-head
  `q_padded`；应让 unified-KV 直接使用 local-head `q_out`，旧 backend 才分配 padding。
- 正式正确性验收需补 output token-id hash、legacy/overlap 逐 token parity、
  256/2048 tokens、BS 1/2/4/8、tier 下降、长短交错及 prefill/decode 混跑。

### INT8 projection GEMV（2026-08-21，未采用）

- CDNA2 `V_DOT4_I32_I8` kernel-only 对三个 shape 分别约快 53%/28%/20%，但完整
  Mori graph 内每层三次合计仍约 37 us，端到端只到约 `59.18 tok/s`，相对当前
  基线约 1%。保持 `SGLANG_DSV4_GFX90A_INT8_WEIGHT_GEMV=0`，不提交为正式收益。

### FP4 MoE KSPLIT / Mori AsyncLL（2026-08-21，进行中）

- `KSPLIT=2` 的纯 AR 256-token 为 `24.24 / 24.23 / 26.78 / 27.26 /
  25.65 tok/s`；`KSPLIT=4` 为 `20.39 / 23.64 / 23.78 tok/s`。两者都确实
  命中 CKTile A16W4，并绕过两次在线 MXFP4 activation quant，不是 selector 假开关。
- `KSPLIT=4` trace 的每层 CK/rocBLAS 类耗时约 `0.38--0.48 ms`、临界层约
  `0.95 ms`，而 ksplit=0 基线整层约 `0.37 ms`。AIter 的 per-1x32 测试在非
  gfx950 上直接跳过；通用 CKTile 小 M kernel 当前不适合作为 gfx90a 正式路径。
- `DEEPEP_MODE=low_latency, MORI_ENABLE_SDMA=0` 的五轮为 `20.66 / 21.08 /
  22.18 / 19.94 / 20.78 tok/s`。trace 显示 AsyncLL CU transfer/wait 每层约
  `0.6--0.9 ms`，上游固定的 `64 blocks x 8 waves` 对 BS1 明显过量。
- 新增默认不改变行为的 AsyncLL geometry 开关：
  `SGLANG_MORI_ASYNCLL_BLOCK_NUM`（默认 64）、
  `SGLANG_MORI_ASYNCLL_WARP_NUM_PER_BLOCK`（默认 8）、
  `SGLANG_MORI_ASYNCLL_RDMA_BLOCK_NUM`（默认 32）。先测 8/16/32 blocks；
  每次 GPU 实验前仍必须先运行 `amd-smi process --general --sort-by-pid -g 4 5 6 7`。
- `low_latency + SDMA`（通用 capacity=256）可到 `40.37 / 47.22 / 49.09 /
  49.54 / 41.31 tok/s`，明显好于无 SDMA，但仍低于 normal Mori。
- AsyncLL transport capacity 直接缩到 16 或 64 都会在 CUDA graph capture 尾部
  `torch.cuda.synchronize()` 永久等待；faulthandler 已确认不是 CPU 编译。该实验
  改用独立、默认关闭的
  `SGLANG_MORI_ASYNCLL_DECODE_MAX_DISPATCH_TOKENS_PER_RANK`，不要复用 normal

### Correctness first-divergence 与 direct FP4 MoE（2026-08-23）

- 官方 France 文件实际包含尾部换行，精确 chat token IDs 为
  `[0,128803,3085,344,270,6102,294,8760,2755,128804,128822]`；无换行版本
  的第 9 个 token 是 `33`，不能拿两者做逐层 oracle。以后必须同时保存 prompt
  `repr` 和完整 input IDs。
- 同输入逐层对比：embedding 完全一致；layer-0 attention 的 Q/core/wo_b 相对误差
  约 `2.63% / 2.15% / 4.67%`，而 MoE 输出跃升到 `14.72%`，整层为 `13.43%`。
- 根因是通用 AIter FP4 fused-MoE ABI 没有传递 DSV4 的
  `swiglu_limit=10.0`。`SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE` 的 gfx90a kernel
  实现 bounded SwiGLU；真实变量名包含 `FP4_`，旧的
  `SGLANG_DSV4_GFX90A_DIRECT_MOE` 是无效假开关。
- EP1 有 256 local experts，不满足当前 direct kernel 的 64-expert layout；EP4/Mori
  正好满足。TP4/EP4、eager、direct-MoE=1 对精确官方 prompt 输出
  `The capital of France is **Paris**.`，9 tokens 后 EOS；generic AIter 路径则从
  首 token 开始进入 `to the capital ...` 循环。
- `scripts/rocm_dsv4_flash.sh` 因此仅在 `EP_SIZE=4` 时默认开启 direct FP4 MoE；
  EP1/EP2 仍默认关闭，并保留显式 0/1 A/B。

### Correctness 修复后的优化逐项恢复（2026-08-23）

- 所有轮次先用精确官方 chat IDs 验证 France，预期为
  `The capital of France is **Paris**.` 并 EOS；性能 harness 使用
  `ignore_eos=true`，保证实际生成固定 256 tokens、finish=length。正确模型会自然
  EOS，旧 harness 不带 ignore_eos 时不再是固定长度基准。
- 正确 EP4/Mori/direct-MoE eager 朴素基线（graph、scheduler/SBO、custom AR、
  MHC/attention/shared compute优化关闭）为 `5.355 tok/s`。
- 仅启用 shared-expert TP 为 `4.853 tok/s`，单独退化；再加入配套 BF16/wave64
  shared compute 后为 `5.560 tok/s`，约 +3.8%，不足单独 checkpoint。
- 在上述 shared 子系统上仅恢复 decode CUDA graph 后为
  `8.984 / 9.125 tok/s`，固定长度 hash 均为 `8bc49f114ad72a74`，相对 eager
  约 +63%，France 正确性保持。CUDA graph 是 correctness 修复后的首个通过优化。
  Mori 的默认 capacity=16。
- 保持稳定 capacity=256，仅设 `SGLANG_MORI_MOE_MAX_INPUT_TOKENS=64` 可把 AIter
  padded-token key 从 1024 降到 64；七轮为 `43.12 / 48.42 / 49.71 / 50.74 /
  50.59 / 50.33 / 49.39 tok/s`，仍不足以替代 normal。前三轮之后 hash 稳定为
  `1d765b3ef2548259`，但早期 hash 不同，必须与 normal 同 harness 对照。

### 小优化叠加与正确性筛选（2026-08-22）

- 正常 Mori 32/32、Sinkhorn=8 的同轮稳定基线约 `55.85--56.79 tok/s`，十轮
  基准以 JSON-packed output token ids 的 SHA256 前 16 位检查正确性；基线 hash
  为 `f3060e252a69f624`。
- `WAVE64_FP32_GEMV=1 + WAVE64_GROUPED_GEMV=1` 十轮 hash 全部一致，稳态约
  `56.79--57.10 tok/s`，净收益不足 1%，可继续作为安全叠加项但不单独提交。
- packed Triton Top-K router 与 native grouped router 都会偶发改变 token hash；前者
  最高约 `56.39 tok/s`，后者约 `54--55 tok/s`，两者均保持默认关闭。
- replicated embedding 十轮 hash 一致，但破坏 scheduler-overlap 的快态稳定性，
  结果在 `47--57 tok/s` 间抖动；保持默认关闭。
- Sinkhorn 8→4 的随机输入 microbench 相对 20 次迭代 comb 最大误差约
  `6.1e-3`，该样本最终 BF16 MHC 输出仍 bitwise exact；端到端 hash 稳定但变为
  `dfdc22ded64b772d`，最高仅约 `57.68 tok/s`，不足以接受该近似。
- `SGLANG_DSV4_USE_BF16_RMSNORM_WEIGHT=1` 最高约 `57.18 tok/s`，但十轮出现多个
  token hash，且慢态明显，不能采用。
- Mori dispatch/combine 24/24 在本轮最高约 `57.40 tok/s`，但偶发 hash 漂移；
  旧轮次接近 `59.7 tok/s` 不能作为稳定结果。低 block geometry 需要先验证 Mori
  barrier/progress 协议，不能作为正式配置。
- AIter FP4 stage-2 的 64-thread/N32 CK symbol 确实存在并命中；新开关
  `SGLANG_DSV4_GFX90A_AITER_MOE_STAGE2_64THREAD` 默认关闭。它将 N CTA 数放大四倍，
  十轮虽 hash 一致但仅 `40--50 tok/s`，明确负收益。
- 保持 Mori 32 blocks，仅将 dispatch 从 8 waves/block 降为 4，十轮 hash 一致，
  峰值约 `57.37 tok/s`，未超过 8-wave + wave64 GEMV 的约 `57.10 tok/s` 到足以采用，
  且仍有 `49.74 tok/s` 慢态；不修改正式默认。
- 曾在 AIter 依赖中把 A16W4 CKTile 的三处 `ksplit > 1` 条件临时放宽到
  `ksplit >= 1`，用于测试无 split reduction、无 activation MXFP4 quant 的路径。
  路径可启动并命中 CKTile，但仅 `22.76--24.98 tok/s`，三轮三个 hash 且输出严重
  错误；条件已完整恢复。这里的 `>1` 是 layout/kernel 契约，不是可直接放宽的假限制。
- FP8 Mori dispatch 对 FP4 routed weights 不是直通：当前 pre-permute 会先
  `FP8→BF16`，AIter 再 `BF16→MXFP4`。短 BS1 下不应只凭 payload 减半就测试；
  需要先实现直接/融合转换，否则很可能是额外双重量化。

### normal/AsyncLL capacity 接线修复与 60 tok/s（2026-08-22）

- 后续 AsyncLL 实验曾把两个 decode capacity 环境变量接反：normal Mori 误读
  `SGLANG_MORI_ASYNCLL_DECODE_MAX_DISPATCH_TOKENS_PER_RANK`，AsyncLL 反而读取
  `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK`。因此脚本设置的 normal
  `capacity=16` 实际失效并退回通用 capacity=256，解释了同一 checkpoint 从约
  `58+` 回落到 `56--57 tok/s`。现已恢复各自正确归属，启动日志必须明确显示
  normal Mori 四 rank 均为 `num_max_dispatch_tokens_per_rank=16`。
- capacity 接线修复后，第一套全新服务的 10 轮纯 AR 为：
  `57.950 / 60.306 / 59.701 / 60.097 / 60.358 / 59.518 / 60.323 /
  60.245 / 60.340 / 53.282 tok/s`，中位约 `60.17 tok/s`；10/10 output-id hash
  均为基线 `f3060e252a69f624`。8 并发复验第二轮 aggregate 为 `230.42 tok/s`，
  全部请求输出 256 token 且 `finish=length`。
- 第二套独立服务的前六轮达到 `60.592 / 60.899 / 58.819 / 59.605 /
  59.635 / 60.876 tok/s`，后四轮为 `59.763 / 59.495 / 58.616 / 58.397`；其中
  一轮 hash 漂移，说明 custom AR/Mori graph 的偶发到达顺序仍需继续做严格
  bitwise 稳定性审计，但 60 tok/s 原生 AR 的性能目标已在独立进程重复越过。
- wave64 attention geometry 离线扫描保持 bitwise exact：grouped `wo_a` 从约
  `18.15` 降到 `16.56 us`；FP32 N=512/1024/2048 分别选择 `(rows,unroll,waves)`
  为 `(1,2,8)/(1,2,4)/(1,2,4)`，相对旧 `(2,2,8)` 约快 20%/持平/7.6%。
- direct BF16-activation/FP4-weight routed-MoE 原型在生产 shape micro-check 中数值
  接近参考，但端到端仅约 `18 tok/s` 且输出 hash 不稳定；逐行 FP4 解码的标量
  wave kernel远慢于 AIter CK，不应作为正式路径。native MHC full/tail、INT8+
  Mori24、INT8+Sinkhorn4 等堆叠也均未超过最终精确配置，继续保持默认关闭。

### TP4/EP1、无 Mori oracle 与 custom AR 正确性修复（2026-08-22）

- SGLang 已有真正的 expert tensor-parallel 权重布局；`TP_SIZE=4 EP_SIZE=1
  MOE_A2A_BACKEND=none` 会让每 rank 持有 256 experts 的 1/4 projection shard，
  实测 FP4 `w13=[256,1024,2048]`、`w2=[256,4096,256]`，显存可容纳。
- gfx90a AIter 本地 tune 原先只接受 EP4/TP1 的 `[64,4096,2048]`。加入
  EP1/TP4 shape 后，现有两阶段 CK FP4 kernel 可以直接运行，无需 Mori。
- 初次 TP-only 结果达到约 `65--70 tok/s`，但偶发 output hash 漂移。关闭 SBO
  和 ROCm multi-stream 后仍复现；切到 RCCL 后 hash 稳定但只有约 `28 tok/s`，
  因而定位为 AIter custom all-reduce 协议问题，而不是 TP 权重/数学错误。
- AIter `cross_device_reduce_1stage` 原先没有 final barrier，快 rank 可以在慢 rank
  仍通过 P2P 读取时复用下一层输入缓冲区；同时 peer signal poll 还是 device scope。
  恢复 final system-scope barrier，并将 signal load 改为 system-scope acquire 后，
  20/20 单请求均得到基线 hash `f3060e252a69f624`。首轮含暖机 `55.82 tok/s`，
  后 19 轮为 `66.11--67.65 tok/s`，稳态中位约 `67.35 tok/s`，相对 60.17
  checkpoint 约提升 12%。配置为纯 AR、TP4/EP1、A2A=none、SBO=off。
- 同一服务并发 native AR：BS2 约 `72.62--73.07 tok/s`，BS4 约
  `138.20--138.82 tok/s`，BS8 约 `245.34--249.58 tok/s`。所有 54 个并发
  请求均生成 256 token、finish=length，且 hash 均为 `f3060e252a69f624`。
  当前 TP-only 的 BS8 也高于 EP4 checkpoint 的 `230.42 tok/s`。
- 修复后同版本 EP4/Mori 的 A/B 已补齐。BS1 十轮 hash 均为基线，但稳态中位约
  `58.1 tok/s`；BS2 约 `67.87--68.70`，BS4 约 `126.53--128.52`，BS8 约
  `229.26--230.61 tok/s`，所有 tier 都低于 TP-only。final barrier 约损失旧 EP4
  峰值的 2 tok/s，但消除了独立服务的偶发漂移，说明旧 60 tok/s 中包含不安全的
  buffer overlap。
- EP4 的 BS4/BS8 虽然同批请求彼此一致，hash 却稳定变为 `1d765b3ef2548259`，
  不等于 BS1/TP-only 的 `f3060e252a69f624`；TP-only 在 BS1/2/4/8 始终保持参考
  hash。因此当前没有证据支持 decode hybrid，反而应先审计 Mori batch-tier 的
  row ordering/quant/combine 数值路径。并发 harness 现在同时报告批内一致性和
  `reference_match`，避免把“全都一致地偏离参考”误判为通过。
- 每次 GPU 实验前继续强制运行：
  `amd-smi process --general --sort-by-pid -g 4 5 6 7`。

### TP4/EP2 hybrid routed MoE（2026-08-22）

- `--ep-size 2` 原先是假配置：通用 A2A post-process 会把 Mori EP 强制扩为
  TP size，实际仍启动 EP4。新增默认关闭的 `SGLANG_MORI_ALLOW_PARTIAL_EP=1`
  后，TP4/EP2 才真正形成两个 expert-TP2 groups 和两组 Mori world-size 2。
- DSV4 A2A source token 必须按 `moe_ep_rank` 切成两块，并在同一 expert-TP
  group 内复制；shared TP4 的最终 global all-reduce 同时完成 shared partial、
  routed TP2 partial 和两个 EP token chunks 的汇合。
- Mori 固定使用全局 rank 0 bootstrap subgroup，导致第二组 `[1,3]` 永久等待；
  修复为将 subgroup rank 0 映射至其 global rank。Mori group registration 也必须
  按 expert-TP lane 使用不同名称，否则两个 subgroup 冲突。
- 实际加载/捕获确认：每 rank `128` local experts、AIter 两阶段 FP4 key 为
  `N=1024, E=128`，四 rank 模型显存均约 `45.34 GB`；Mori 两组都为
  `world_size=2`，graph BS1/2/4/8 全部捕获成功。
- 32-block、SBO off 的 BS1 稳态中位约 `59.93 tok/s`；SBO+多流后约
  `60.77 tok/s`。将 world-size-2 dispatch/combine 都降为16 blocks 后，后六轮
  稳定在 `62.22--62.31 tok/s`，10/10 hash 为 `f3060e252a69f624`。仍低于
  TP4/EP1 的 `67.35 tok/s`，不应作为 BS1 默认。
- 16-block EP2 并发结果：BS2 `77.17--78.11 tok/s`（hash 为 BS1 reference），
  BS4 快态 `143.74--145.68`、一轮慢态 `123.60`，BS8 `249.37--256.51`、
  中位约 `253.5 tok/s`。相对 TP-only，BS2 约提升7.4%，BS4/BS8小幅领先；
  相对 EP4 BS8 的约230提升约10%。因此 EP2是并发候选而非单请求候选。
- BS4/BS8 稳定得到 tier hash `1d765b3ef2548259`；其中混有既有的 batch-tier
  MHC/Sinkhorn和归约顺序差异。批内所有请求一致，但在建立同-tier reference
  oracle 前不能宣称与 BS1 bitwise parity。

### TP4/EP2 graph BS16 大并发（2026-08-22）

- 为捕获真实 tier 16，而不是让16请求退回 eager，使用：
  `CUDA_GRAPH_MAX_BS_DECODE=16`、
  `AITER_GFX90A_MXFP4_QUANT_MAX_ROWS=128`、
  `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK=32`；其余保持 EP2、
  Mori dispatch/combine 16 blocks、SBO+多流。最终 graph tiers 为
  `[1,2,4,8,12,16]`，注册1044个graph地址，capture后每 rank仍有约5.6 GB余量。
- 同一新服务 BS8 五轮 aggregate 为 `227.04 / 235.32 / 246.35 / 253.07 /
  253.62 tok/s`；前几轮有明显升温，后两轮稳定约253.3，与 graph-max-8 的
  EP2结果一致。
- BS16 五轮 aggregate 为 `350.52 / 356.63 / 340.12 / 335.27 /
  360.93 tok/s`，中位约 `350.52 tok/s`。每轮16/16请求都输出256 token、
  finish=length，批内hash全部一致为 `1d765b3ef2548259`。
- 并发 harness 使用独立 cache salt；page-size=256 的prefill按请求逐步 admission，
  server log显示 running requests 从0/1逐渐爬升到15。因此上述是严格HTTP group
  wall-time aggregate，不是所有16请求从第一个decode step起就同时驻留的纯kernel
  BS16上限。不同请求出现约22.6与25.9 tok/s两组完成时间，也来自该入场先后。

### TP4/EP2 + DSpark 并发常驻与显存边界（2026-08-22）

- DSpark 使用 checkpoint 自带 draft，`gamma=5`、每次 target verify 最多6 token；
  服务固定为 TP4/EP2、两组 Mori world-size2、dispatch/combine 16 blocks，GPU 4--7。
  DSpark 测速必须使用独立 `start-dspark` / `bench-dspark-concurrent` 命令，不能移除
  原生 AR harness 对 `SPECULATIVE_*` 的硬拒绝。
- DSV4 indexer 原先只允许 decode/idle 读取 CUDA graph 的 dense/sparse capture variant；
  DSpark 的 `TARGET_VERIFY` 会在此前提前返回，导致所谓 dense graph 仍分配完整
  FP32 indexer logits，96-token tier 单次尝试额外分配578 MiB并 OOM。修复后
  `TARGET_VERIFY` 也按 capture variant 选择；新增默认关闭的 dense-only graph 模式，
  仅适用于 `max_kv_len <= index_topk=512` 的短上下文服务。
- DSpark target verify 的 graph key 是展平 verify token 数，而不是请求数：BS16需要
  96-token target graph。96 tier 可绕过 indexer但在 `M=576` Mori/FP4 capture 中
  无法稳定完成；48 tier 的 target graph可以完成，但约占7 GiB，随后 draft graph
  没有余量。因此最终常驻使用 graph max16：完整 graph覆盖BS1/BS2，BS4/8/16的
  target verify为 eager。
- `pre_warm_nccl` 的说明称 AMD 默认开启，但当前 dataclass/default 实际为 false。
  未预热时 graph 后只剩约0.11 GiB，首请求 RCCL all-gather 的2--6 MiB动态分配即
  OOM。DSpark命令现显式添加 `--pre-warm-nccl`，使持久通信分配进入内存预算。
- 为同时容纳 target graph、draft graph和预热RCCL，最终关闭约1.1 GiB/rank的
  `SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR` cache；这是先前端到端约1%的小优化，
  shared expert和routed MoE关键路径仍保留。最终 target graph约6.825 GiB、draft
  graph约0.322 GiB，startup可用显存约0.59--0.70 GiB，token capacity仍为8192。
- 最终常驻参数要点：`mem_fraction_static=0.85`、graph tiers
  `[1,2,4,6,8,12,16]`、Mori capacity=256、decode capacity=64、MXFP4 quant rows=128、
  attention BF16 cache off、RCCL prewarm on。服务PID为1093654，日志为
  `/tmp/sglang_dsv4_flash_dspark_ep2_resident.log`。
- 256-token、独立HTTP请求、每档3轮的aggregate DSpark吞吐如下；数字按客户端实际
  收到的token/group wall time计算，不是原生AR forward-per-token口径：
  - BS1：`58.245 / 61.555 / 61.808`，中位 `61.555 tok/s`；
  - BS2：`108.398 / 115.751 / 97.218`，中位 `108.398 tok/s`；
  - BS4：`167.794 / 194.503 / 195.059`，中位 `194.503 tok/s`；
  - BS8：`275.183 / 257.585 / 270.077`，中位 `270.077 tok/s`；
  - BS16：`288.454 / 296.673 / 311.179`，中位 `296.673 tok/s`。
- 同prompt接受统计：BS1平均接受长度约4.339、接受率约0.678；BS2/4约4.414、
  约0.693；BS8约4.339--4.414；BS16约4.197--4.414、约0.646--0.693。
  所有正式请求均生成256 token、finish=length，且各BS的output-id hash均为
  原生AR参考 `f3060e252a69f624`。
- 与同机native EP2中位数相比，DSpark约为：BS1持平，BS2 +39%，BS4 +34%，
  BS8 +6.5%，BS16 -15%。BS16 native graph仍更强，说明高并发下固定6-token verify、
  draft开销及target eager抵消接受收益；当前DSpark甜点区在BS2--BS8。

### gfx90a CKTile A16W4 stage-2 W2 行置换修复（2026-08-23）

- 旧的约60 tok/s AIter FP4路径并非数值正确：legacy CK FP4xFP4 stage1在gfx90a
  输出全零；改走CKTile BF16×FP4后，stage1相对direct oracle的cosine约
  `0.999996`，但stage2只有约`0.24`。因此基础错误不是Mori、attention、MHC或
  `tid2eid`，而在routed expert的W2阶段。
- 放开AIter自带的A16W4 reference测试后，单卡、`topk=1`、无padding也稳定复现
  约80%输出维错误；生产shape `M=1, N=4096, K=2048`同样失败。这排除了top-k
  atomic combine、padding和SGLang wrapper。
- 将W2 E8M0 scale全部设为同一值仍失败，排除scale permutation。用16个token给
  每个输出列生成指纹后，256/256列均可与reference列以`0.99996--1.0` cosine
  唯一匹配；每个128维N tile内的16-row block映射为：
  `fast -> reference = [0,2,4,6,1,3,5,7]`。
- 在现有A16W4 preshuffle前，对raw W2权重和W2 scale应用逆置换
  `[0,4,1,5,2,6,3,7]`，即可恢复逻辑输出顺序；这是加载期一次性重排，不增加
  decode kernel或运行时开销。独立测试在`4096x2048`的topk=1/topk=6下不再
  触发`logits_diff > 1e-3`。
- 固定France输入IDs：
  `[0,128803,3085,344,270,6102,294,8760,2755,128804,128822]`。完整
  TP4/EP4+Mori eager输出10/10为
  `[671,6102,294,8760,344,2619,51119,42499,1]`，文本为
  `The capital of France is **Paris**.`，completion hash唯一值
  `6f41fe2f01d52507`。CUDA graph仅捕获BS1后再次10/10完全一致，graph capture
  约7秒，9-token问答稳态约0.69秒。
- 每次GPU实验前使用：
  `/opt/rocm/core-7.14/bin/amd-smi process --general --sort-by-pid -g 0 1 2 3`。

### W2 修复后的真实 TP/EP 基线与 shared-expert overlap（2026-08-23）

- W2 行置换修复后，旧的约 `60 tok/s` 不再是有效正确性基线：旧 legacy CK
  FP4 stage-1 实际输出全零，等价于大部分 routed expert 计算缺失。正确的
  TP4/EP4+Mori、graph BS1、scheduler overlap+SBO 基线仅约
  `14.2--15.0 tok/s`。
- `SGLANG_MORI_DECODE_TIERED_CAPACITY=1` 确实把 AIter 输入 padding 从64行降至
  16/8行，但 CKTile kernel 的最小 `block_m=32` 没有变化，端到端仅约
  `14.5--15.3 tok/s`，随后短请求还出现一次挂起；不应作为默认开关。
- 真正的 TP4/EP1/no-A2A 对照已运行成功：graph BS1、scheduler overlap on、
  SBO off、custom all-reduce off，France 固定 token oracle 正确；256-token五轮为
  `20.169 / 20.142 / 19.759 / 20.145 / 20.000 tok/s`。相对正确 EP4 基线约
  提升34%，证明 BS1 的 EP4/Mori 固定税显著，但 routed CKTile 小 M 仍是主瓶颈。
- TP4/EP1 开启 SBO、但 ROCm 未创建 `alt_stream` 时，shared-expert pre-combine
  hook 会对 `None` 调用 `wait_stream`，在 graph capture 崩溃。构造时现在只有在
  `alt_stream is not None` 时才注册这组 SBO hook；否则保留普通顺序路径。
- 显式启用 `SGLANG_ROCM_USE_MULTI_STREAM=1` 后，TP4/EP1 + SBO + graph BS1 的
  France token IDs 仍与 oracle 完全一致，但256-token五轮仅为
  `19.328 / 19.870 / 19.895 / 19.753 / 19.789 tok/s`，中位约19.79，较SBO-off
  基线约慢1.7%。因此 multi-stream shared/routed overlap 不作为默认性能配置；
  应直接优化 `M<=8` 的 BF16-activation/FP4-weight routed expert kernel。

### direct FP4 wave64 的 TP4/EP1 探针（2026-08-23）

- W2 CKTile加载期补偿原先会作用于所有gfx90a AIter FP4 MoE，现已严格限制到
  DSV4三组raw packed layout：EP4
  `w13=(64,4096,2048), w2=(64,4096,1024)`；EP2
  `(128,2048,2048),(128,4096,512)`；EP1
  `(256,1024,2048),(256,4096,256)`。TP4/EP1 graph BS1再次得到France固定
  token oracle，输出IDs仍为
  `[671,6102,294,8760,344,2619,51119,42499,1]`。
- no-A2A standard dispatcher的`expert_mask`和`num_local_tokens`都为None。direct
  开关此前在这种情况下会eligibility miss，却继续把raw权重静默交给要求preshuffle
  的CKTile，产生乱码。direct kernel现有static-live/no-mask入口；raw权重contract
  miss时直接报错，不再静默回落。
- 将direct FP4 GEMV放开至EP1后，原标量实现64-token稳态约
  `13.64--13.86 tok/s`。改用CDNA2 FP16 `amd_mixed_dot`/FP32 accumulation后为
  `14.19--14.79 tok/s`，约提升7.8%，France oracle不变。
- TP4/EP1 down projection的K=512只有16个scale groups；把wave64拆为四个
  16-lane subgroup并行处理Top-6 slots，router weight在子组归约后以FP32施加。
  64-token稳态进一步到`17.42--17.53 tok/s`，相对原标量direct约+28%，France
  oracle仍精确一致，但仍低于correct CKTile约20 tok/s。
- `kRows=4`降至约`16.1 tok/s`，down blocks 256→512没有实质收益，均撤回。
  EP4/K=2048使用subgroup direct只有约`9.2--10.7 tok/s`，虽然France正确但明显
  慢于CKTile。故`SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE`统一默认关闭，只作为小M
  kernel研究路径；正式路径继续使用W2顺序已修复的CKTile A16W4。
- 当前A16W4 CKTile实际decode tile是M16，不是SGLang tune字典中误导性的32；
  真正M4方案需扩展CK mixed-FP4 policy至gfx90a已有的BF16
  `M4N64K16` MFMA，并同步实现NLane64 weight/scale shuffle及新的W2列序oracle。

### TP4/EP1 packed-FP4 × INT8-dot decode（2026-08-23）

- direct kernel的剩余热点不是HBM：BS1把activation协作搬到LDS后，gate/up仅从
  约190.2降到187.8 us，down从114.5降到111.2 us。核心限制是FP4解包后仍做
  FP16/scalar dot。
- gfx90a/CDNA2没有原生FP4 MFMA，但有`v_dot4_i32_i8`。每个32元素group现在由
  一个block线程只量化一次BF16 activation为INT8；E2M1 FP4 nibble精确映射为
  `0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12`，INT32 dot结果再乘
  `activation_scale * E8M0_scale * 0.5`并FP32累加。没有离线展开权重，也不增加
  常驻VRAM。
- TP4/EP1真实shape单卡microbenchmark：gate/up约`45.90 us`，down约
  `32.11 us`，两段约`78.69 us`；相对LDS+FP16-dot的约299 us提升约3.8倍。
- 完整TP4/EP1、no-A2A、graph BS1、scheduler overlap on、SBO off、custom AR off
  的256-token native AR稳态为：`25.649 / 24.563 / 25.550 / 25.540 /
  25.050 tok/s`，中位约`25.54 tok/s`。相对correct CKTile约20.14提升约26.8%。
- 6轮256-token输出hash全部为`9db479653c78d5fc`；France固定输入10/10均为
  completion hash`6f41fe2f01d52507`及完全相同的9个token。urllib出现的502来自
  客户端继承HTTP代理；禁用ProxyHandler后10/10正常，服务health始终200。
- EP4使用同一subgroup direct路径只有约`9.2--10.7 tok/s`，所以脚本仅在
  `EP_SIZE=1`时默认开启direct INT8-dot；EP2/EP4继续使用correct CKTile。
- 新direct基线上重新加回shared/routed multi-stream+SBO，中位约`25.21 tok/s`；
  仅开SBO中位约`25.06 tok/s`，都低于SBO-off的约25.54。TP4/EP1默认因此明确
  关闭SBO；EP2/EP4保留原默认。
- 重新启用现有gfx90a peer-read custom all-reduce后，TP4/EP1通信固定延迟大幅
  消失：256-token六轮为`59.873 / 60.197 / 60.199 / 60.209 / 60.204 /
  60.199 tok/s`，中位约`60.20 tok/s`，相对RCCL基线约25.54提升约136%。六轮
  output hash均为`4690817e29438b74`；France固定输入另测10/10均为
  `6f41fe2f01d52507`。这是当前首次同时满足correct routed MoE与60 tok/s目标的
  native AR checkpoint。
- 最终TP4/EP1默认组合：direct packed-FP4×INT8-dot开启、custom all-reduce开启、
  scheduler overlap开启、SBO关闭、multi-stream关闭、graph BS1。若要做RCCL
  correctness A/B，显式设置`DISABLE_CUSTOM_ALL_REDUCE=1`。

### TP-only direct-MoE / MHC geometry ABBA（2026-08-23）

- 固定前四个GCD、TP4/EP1、no-A2A、graph BS1、native AR及
  `4690817e29438b74`正确性hash。原direct kernel为`256 blocks x 4 waves`；
  `128 blocks x 8 waves`明显退化，而`208 blocks x 8 waves`对应MI250X每个
  104-CU GCD恰好两block/CU。真实shape microbench中gate/up约从147.2降至
  136.9 us、down从102.3降至97.7 us。
- 仅direct几何的两套独立服务trimmed median约`61.31 / 61.09 tok/s`；返回A2
  为`59.44 tok/s`，端到端单变量约`+2.8--3.1%`，不足单独checkpoint。
- MHC原`block_n=4`的48-CTA split-K几何是为Mori progress让CU；TP-only没有
  Mori，因此使用`block_n=1`的192-CTA scalar-row。单变量trimmed median
  `60.44 tok/s`，相对A2约`+1.7%`，hash不变。新开关
  `SGLANG_DSV4_GFX90A_MHC_TP_ONLY_GEOMETRY`只在脚本`EP_SIZE=1`时默认开启，
  EP2/EP4继续保留48 CTA。
- 两项叠加后B1六轮trimmed median约`62.70 tok/s`；B2独立重启六轮为
  `62.427 / 63.208 / 63.383 / 62.822 / 63.428 / 63.465 tok/s`，trimmed
  median约`63.30 tok/s`。返回A3六轮trimmed median约`57.53 tok/s`；即使以
  较高A2比较，B2仍约`+6.5%`。所有B轮长输出hash一致，France固定IDs也保持
  精确输出`The capital of France is **Paris**.`。
- 最终代码第三次独立启动的六轮为`58.408 / 62.475 / 62.867 / 63.274 /
  61.955 / 63.461 tok/s`；首轮冷态后trimmed median约`62.67 tok/s`，hash仍
  6/6一致。脚本只在`EP_SIZE=1 && MOE_A2A_BACKEND=none`时默认启用192-CTA
  MHC几何。
- `AITER_GFX90A_AR_SMALL_BLOCKS=2`单变量trimmed median仅`58.50 tok/s`，低于
  默认4 CTA的A2，证伪；不改默认。
- 尝试接线AIter中尚未使用的`fused_allreduce_mhc_post`。修正实验kernel的comb
  转置后France仍正确，但256-token hash改变且首轮漂移，trimmed median约
  `58.2 tok/s`，正确性和性能均未通过；SGLang接线与依赖实验改动均撤回。
- AIter JIT必须显式使用`/opt/rocm/core-7.14/bin/hipcc`并设置ROCm include/lib；
  conda的`/home/pc/anaconda3/bin/hipcc`会落到缺少hipsparse/thrust头的host工具链。
- 继续扫描MHC split-K：TP-only `block_n=1`下，tail microbench的9轮trimmed
  median为split4 `101.89 us`、split8 `95.54 us`、split16 `102.17 us`；8-way
  是清晰局部最优。direct MoE top-6真实shape下，208 blocks的gate/down为
  `135.25/93.25 us`，192为`135.49/93.17 us`，224退化到
  `158.80/96.28 us`，因此保持208。
- 尝试让direct row-major FP4权重同时使用row-major E8M0 scale，省去CK scale
  swizzle地址计算；microbench gate/down合计约快`1.75%`，数值oracle cosine为
  `0.999974/0.999967`。但完整服务B的8轮trimmed median约`62.63 tok/s`，返回
  committed A约`62.88 tok/s`，长hash均为`4690817e29438b74`，端到端无收益，
  因此代码已撤回。

### TP-only native router GEMV（2026-08-23）

- dense-only graph trace确认短上下文已正确跳过full indexer；此前capture trace里
  每token约`2.27 ms`的indexer logits来自dual-graph最后捕获的sparse variant，
  不是短请求实际replay瓶颈。dense trace中router projection为每层一次
  `[1,4096] x [4096,256]` Tensile GEMM，约`19.4 us`，后接约`14.1 us` router
  Triton。
- 将已有gfx90a wave64 BF16 GEMV扩展到N=256。micro中
  `rows=1, unroll=2, waves=4`的11轮trimmed median约`12.92 us`，但完整graph
  仅约`64.84 tok/s`；8-wave虽standalone约`13.50 us`，端到端却稳定更快，说明
  router workgroup数量会与同graph的其它kernel/collective发生资源竞争，最终保留
  `rows=1, unroll=2, waves=8`。随机权重及真实checkpoint router权重上与
  AIter/tgemm均为BF16等价数学；真实权重的20组输入中每5120个logits仅1--4个
  元素因归约顺序相差约1 ULP。
- 完整TP4/EP1/no-A2A/graph-BS1两套独立B服务的8轮trimmed median分别约
  `66.17 / 66.08 tok/s`，返回A约`62.88 tok/s`，提升约`5.1%`。France固定IDs
  仍精确输出`The capital of France is **Paris**.`；256-token hash在所有B轮稳定为
  `0569e30b92219d8c`。它与旧hash不同，原因是合法BF16 reduction-order差异而非
  随机竞态：两次2048-token生成均完整结束且hash同为`8845939925bbe186`，速度
  `49.07 / 47.75 tok/s`。前三个hash-router层单独回退AIter不改变该新hash且略慢，
  因此最终对全部M=1 DSV4 router使用native路径；prefill及其它shape保持AIter。
- 在native router checkpoint上重新测试旧的packed-key
  `SGLANG_DSV4_GFX90A_TRITON_TOPK_ROUTER=1`：standalone wrapper从约
  `68.24 us`降到`37.45 us`且随机输入IDs/weights逐元素一致，但完整graph的8轮
  trimmed median只有约`65.40 tok/s`，低于generic Top-K的约`66.1`，并产生稳定的
  另一条hash `35780d61502a7170`，因此继续默认关闭。
- router GEMV每wave多行复用也只在micro胜出：`rows=4,waves=8`约`13.02 us`，
  但完整graph trimmed median约`64.99 tok/s`；最终保持`rows=1,waves=8`。这再次
  说明gfx90a graph需要按全局workgroup供给与collective交错选择geometry，不能只按
  standalone kernel时间定默认值。

### TP-only routed-MoE prefill INT8 与 expert 分组（2026-08-24）

- 原raw-FP4 direct kernel把M=256 prefill走入逐assignment的FP16 dot分支；
  1028-token prompt稳态TTFT约`12.85 / 12.95 s`，4604-token约
  `58.01 / 58.32 s`。不能直接回退AIter CKTile，因为direct decode保留的是raw
  packed权重，而CKTile要求preshuffle；将raw权重静默传入会得到错误结果，完整复制
  两套expert权重又超出可接受VRAM。
- 对M>1先用group-32 INT8量化activation，再由CDNA2 `v_dot4_i32_i8`消费raw FP4
  codebook；gate/up单层约`43.51 -> 8.65 ms`，down约`27.85 -> 6.87 ms`。
  1028-token稳态TTFT降到`3.15 / 3.41 s`，4604-token降到
  `14.00 / 14.10 s`，均约4倍；M=1 decode保持原共享量化路径。
- `CHUNKED_PREFILL_SIZE=512`对1028-token无稳定收益，对4604-token仅从约
  `14.05`降到`13.56 s`（约3.6%），说明主要成本不是scheduler chunk次数。
- 新开关`SGLANG_DSV4_GFX90A_FP4_GROUPED_PREFILL`使用AIter sorter的编码
  （低24位token、高8位top-k slot），每wave让同expert的2个assignment复用一次
  raw FP4 weight/scale读取。真实TP4/EP1 shape micro中gate/up为
  `8.63 -> 7.71 ms`且BF16输出逐元素一致；group4约7.78 ms，group8退化到9.34 ms。
- chunk=256端到端：1028-token稳态TTFT`3.049 / 3.014 s`，相对未分组中位
  `3.283 s`提升约8.3%；4604-token为`13.200 / 13.174 s`，相对`14.045 s`
  提升约6.5%。固定France输入5/5精确输出
  `The capital of France is **Paris**.`并EOS。256-token decode稳态
  `65.40--65.55 tok/s`且6轮hash一致；grouped路径仅作用于M>1。
- 同一sort metadata继续用于grouped down：16-lane subgroup保持旧kernel的K归约
  形状，FP32 per-slot partial再按固定top-k顺序归约，避免atomic非确定性。真实shape
  micro为`6.95 -> 4.24--4.31 ms`（约39%）；group4/group8约4.42 ms，均慢于
  group2。相对旧kernel仅约14--22/1048576个BF16元素因最后加法/存取边界相差，
  最大差异0.0078125--0.0625。
- grouped gate+down完整服务的1028-token稳态TTFT为`2.597 / 2.530 s`，比
  gate-only再快约15.4%，相对原始约5.0倍；4604-token为`11.181 / 11.042 s`，
  比gate-only再快约18.7%，相对原始约5.2倍。France固定IDs再次5/5精确；
  256-token长生成4轮hash一致，稳态`63.93--64.70 tok/s`。该长轨迹hash因
  prefill INT8及极小down归约差异改变，不能与旧FP16 prefill hash直接等同。
- 进一步发现首版grouped kernel只复用了weight地址/L1，但每个assignment仍重复
  8次FP4 nibble到INT8的codebook解包。现在每个32-weight group先解包成8个
  packed INT8 word，再复用于两个activation的`v_dot4_i32_i8`。真实shape micro：
  gate/up `8.63 -> 4.65 ms`（约46%），down `6.95 -> 2.83 ms`（约59%）；gate
  输出bitwise exact，down约12/1048576个BF16元素有极小归约边界差异。
- 完整服务1028-token稳态TTFT为`1.845 / 1.776 s`，相对上个checkpoint再提升
  约29.4%、相对最初约7.1倍；4604-token为`7.741 / 8.142 s`，中位`7.942 s`，
  相对上个checkpoint再提升约28.5%、相对最初约7.3倍。France固定IDs 5/5精确；
  256-token hash稳定为`51e2ac132057ead3`，稳态约`64.24--65.90 tok/s`。
- 在解包复用后重新扫描chunk：512的1028-token TTFT为`1.735 / 1.652 s`
  （中位1.694，较256约+6.5%），4604-token为`7.052 / 7.294 s`
  （中位7.173，较256约+9.7%）。1024反而为1.731/7.281秒，均略慢于512；
  因此gfx90a DSV4脚本默认chunk由256改为512，不继续放大。
- 1024-token单chunk profiler显示43层group2 gate/down分别约0.72/0.42秒，
  routed MoE仍占GPU时间约73%；attention约66 ms、dense FP8约60 ms、peer
  collective约39 ms，暂不应优先。
- shared-unpack后重新扫expert group，曲线与旧实现不同：真实M256 gate的
  group2/4/8约`4.61 / 3.11 / 2.54 ms`，down约`2.80 / 2.03 / 1.61 ms`；
  group16的gate因padding/寄存器压力回退到2.69 ms，故统一采用group8并复用一份
  sorter metadata。
- group8+chunk512完整服务：1028-token稳态TTFT`1.105 / 1.030 s`，中位
  `1.068 s`，较group2再快约37%；4604-token为`4.243 / 4.194 s`，中位
  `4.219 s`，再快约41%，相对最初58.16秒约13.8倍。France固定IDs 5/5精确；
  256-token hash 5/5稳定为`51e2ac132057ead3`，稳态`66.16--66.28 tok/s`。
- M512下group16的micro合计比group8约快10.7%，但完整服务仅约2--3%，不足
  checkpoint，仍统一保留group8。新profile中M512的43层gate/down约162/107 ms，
  合计仍占GPU时间约67%。
- 参考CDNA2 ISA及CK的寄存器LUT做法，将每4个E2M1 nibble的switch解包改成三次
  `v_perm_b32`：正/负8-entry byte LUT各一次，最后按sign selector合并。全部65536种
  uint16组合与原codebook穷举完全一致。group8 M256 micro中gate约
  `2.54 -> 1.91 ms`、down约`1.61 -> 1.44 ms`。
- 完整服务4604-token TTFT从`4.219 -> 3.838 s`（约+9.0%）；France固定IDs
  5/5精确。256-token native AR除首轮JIT外为`74.50 / 72.07 / 74.39 /
  68.63 / 74.08 / 74.53 / 73.88 tok/s`，中位约74.08，较66.2约+11.9%；
  7/7 hash保持`51e2ac132057ead3`，证明新的unpack位映射没有改变数学结果。
- group8 launch几何按M512/M256扫描：gate在416 blocks×8 waves最优，down在
  312 blocks×8 waves最优；小于128行仍保留208 blocks。ABBA中A(208/208)的
  1028-token稳态中位约1.018秒，B1约0.918秒，返回A约1.018秒，B2约0.931秒，
  中等prefill稳定提升约8.5--9.8%；4604-token收益随其它瓶颈占比上升而缩到
  约0.5--3%。B2 France 3/3精确，短decode `74.52 / 74.63 tok/s`且hash不变。
- BF16 down partial把每层临时流量减半，但相对FP32 grouped down仅约0.8% micro
  收益，同时约79万/209万元素发生BF16级变化；已回退，继续使用FP32 partial。
- 实验性full prefill CUDA Graph只捕获512 tier：补齐EXTEND bucket及graph state后
  capture可在约3.6秒完成且无OOM，但replay先因累计seq_len导致page-table shape
  `2 -> 4`失败；固定为最大page-table shape后，`_prefill_lengths_kernel`读取失效
  metadata地址并触发GPU memory fault。该路径已完整撤回；要继续必须先让所有
  prefill length/page/compressor metadata使用runner-owned capture-stable buffer。
- 在group8+`v_perm`+大网格后重新测试chunk1024。真实M1024 micro中group8仍优于
  group16/32，gate/down约`6.27 / 4.49 ms`，比两个M512 chunk合计约省6%。完整
  1028-token稳态TTFT为`0.900 / 0.897 / 0.924 / 0.928 s`，与chunk512接近；
  4604-token为`3.591 / 3.561 / 3.561 s`，相对chunk512 B2中位约3.815秒提升
  约6.7%。France 3/3精确，256-token hash不变且decode约74.2--74.6 tok/s；
  因此脚本默认chunk从512更新为1024。
- 实测CDNA2 `v_mfma_i32_16x16x16_i8`的lane布局：输入按
  `matrix_index=lane&15, k_lane=(lane>>4)*4`装载时精确计算`A @ B.T`；输出
  每lane的4个VGPR对应`row=(lane>>4)*4+[0..3], col=lane&15`。12组随机输入的
  256个输出寄存器均与CPU int32 reference一一匹配。
- 基于该布局实现过group16、K-split 1/2/4/8的FP4×INT8 MFMA探针。公平
  `M=16,N=2048,K=4096`下，split4/8约`0.326 ms`，只比正式group8风格sdot
  基线`0.336 ms`快约3%；split1反而约`0.405 ms`。接入真实swizzled scale后，
  gate/up约`0.305 -> 0.293 ms`（+4%），但down约`0.169 -> 0.187 ms`
  （-11%）；旧sdot down改group16更退化到约`0.223 ms`。MFMA与旧输出相对L2
  误差分别约`2.4e-5 / 8.7e-6`，布局正确但预计整模型净倒退。所有接线和探针均
  已撤回；结论是不能只替换dot指令，下一代版本需同时重做scale布局或融合更大
  expert阶段，才能摊薄每32元素一次的scale转换与LDS归约。
- MI250X的`N=8192,K=1024` block-FP8配置此前只覆盖`M<=128`，chunk1024
  会错误沿用`BM16/BN64/group1/4 waves`小M配置。补测M512/1024后采用
  `BM64/BN64/BK128/group8/4 waves`：对应micro分别从约
  `1.53 -> 0.946 ms`和`2.94 -> 1.72 ms`（+38%/+41%），四种候选输出均与
  旧配置逐元素bitwise exact。
- 完整TP4/EP1 no-A2A ABBA（每服务去掉首轮JIT）：1028-token A合并稳态中位
  约`0.942 s`，B合并约`0.889 s`，TTFT提升约5.6%；4604-token约
  `3.598 -> 3.559 s`，仅约1.1%，说明长上下文的剩余时间主要不在该shape。
  B路径256-token hash仍为`51e2ac132057ead3`。服务日志同时暴露每层
  `M=1024,N=256,K=4096` BF16 GEMM未命中AIter tuned config并退回torch，
  是下一项prefill配置/融合目标。
- 同样补齐其余四个MI250X block-FP8 shape的M512/1024配置，统一使用
  `BM64/BN64/BK128/group8/4 waves`。M1024 micro：`1536x4096`
  `8.45 -> 1.37 ms`、`4096x2048` `2.93 -> 1.67 ms`、`4096x4096`
  `5.74 -> 3.15 ms`、`4096x512` `0.82 -> 0.58 ms`；M512分别约
  `4.27 -> 0.77`、`1.51 -> 0.92`、`2.92 -> 1.66`、`0.45 -> 0.38 ms`。
  所有shape的新旧输出均逐元素bitwise exact。
- 完整服务第二批稳态复测：1028-token连续8轮为`0.870--0.883 s`、中位约
  `0.876 s`，相对全旧A约0.942秒提升约7%；4604-token后7轮为
  `3.441--3.475 s`、中位约`3.446 s`，相对全旧A约3.598秒提升约4.2%。
  后四shape在第一批`8192x1024`之上主要改善长上下文，单独增益约3%；decode
  hash继续保持`51e2ac132057ead3`。
- 为完整1024-token chunk新增group32、split-K=4的CDNA2 INT8 MFMA routed-MoE。
  真实TP4/EP1 shape（E256、topk6、I512）M1024 micro中gate/up从约
  `6.27 -> 3.98 ms`，down从`4.46 -> 3.71 ms`，合计提升约28.3%；M512合计
  反而慢约4.6%，所以selector严格限制`M>=1024`，尾块仍走group8 sdot。
  真实swizzled FP4 scale/layout oracle相对旧实现的L2误差约
  `1.9e-5 / 2.5e-5`，分别只有约1930/390个BF16输出因FP32归约顺序不同。
- GCD0--3严格返回A对照：A的1028-token十轮中位约`0.881 s`，4604-token八轮
  中位约`3.445 s`；B分别约`0.757 s`和`2.924 s`，prefill throughput提升约
  `16.4% / 17.8%`。B的4604+512两轮均完整`finish=length`，TTFT约2.918秒、
  decode约46.2 tok/s。
- 4604+512 completion hash在A和B内都会从第0个生成token分叉；A的独立两轮也
  分别从token0走入不同greedy轨迹，因此该重复`indexer`压力prompt不是bitwise
  correctness oracle，不能把漂移归因于MFMA。短prompt的既有稳定hash仍需继续
  保留；长prefill后更严格的验收应改用teacher-forced logits/固定continuation。
- 测试期间node0物理地址`0x3f87cc...`连续触发host DRAM uncorrectable MCE，内核
  向scheduler发送SIGBUS；这不是kernel或权限问题。地址属于NUMA node0，使用
  `numactl --membind=1 --cpunodebind=1`后服务加载及全部A/B稳定。硬件坏页应在
  维护窗口重启并检查DIMM/EDAC，测试前继续用`amd-smi process`确认资源。
- MFMA32下投影继续扫描K-split/网格：原split4/312约`3.707 ms`，split2在
  624/832/1040 blocks分别约`3.391/3.296/3.294 ms`，1248又退到3.64ms；采用
  split2/1040。严格ABBA中旧几何A约`0.750 s / 2.921 s`（1028/4604 tokens），
  B约`0.727 s / 2.851 s`，返回A约`0.745 s / 2.915 s`，稳定再提升约3.2%/2.4%。
  最终B的4604+512第二轮TTFT 2.851秒、decode 46.07 tok/s并完整finish=length；
  短decode去JIT后`72.61--72.70 tok/s`且3/3 hash仍为`51e2ac132057ead3`。
- M1024 MFMA profile的完整TP0 kernel sum约580ms：gate/up 155.7ms、down136.7ms、
  sparse prefill attention 66.3ms、AIter peer all-reduce 43.5ms、dense FP8 34.1ms、
  MHC约49.7ms、INT8 quant 12.4ms、down reduce 3.8ms、sort约2.2ms。目标2k tok/s
  要求约512ms/chunk，仍需减少约68ms；下一优先级为attention及collective。
- AIter BF16 `M1024,N256,K4096`探测中，默认JIT误选conda的
  `/home/pc/anaconda3/bin/hipcc`并缺`thrust/complex.h`；显式使用
  `/opt/rocm/core-7.14/bin/hipcc`后可编译，但gfx90a没有BF16 ASM heuristic
  kernel。2226个hipBLASLt solution最快一轮约83.5us，对同轮torch约91.7us仅
  9%，而独立稳定torch测量约43--44us，因此不接入该不稳定候选；AOCC不适用于
  HIP device kernel编译。
- DSV4 sparse paged-prefill attention在gfx90a的`H=16,D=512,BLOCK_K=16`
  tile原先发射8个wave，但实际4个wave已覆盖整块。代表性T1024、prefix512、
  extend128 micro从约`4.902 -> 2.478 ms`，输出逐元素bitwise exact；BLOCK_K
  32/64更慢且引入约1e-3相对误差，未采用。严格ABBA中A(8-wave)的1028/4604
  token中位约`0.733 / 2.852 s`，B(4-wave)约`0.694 / 2.608 s`，返回A约
  `0.733 / 2.852 s`，即约5.6%/9.4%收益。B的两轮4604+512均完整
  `finish=length`，TTFT约2.606--2.608秒、decode约46.34 tok/s；长重复prompt
  的completion仍不作为bitwise oracle。
- MFMA down几何与4-wave sparse attention叠加后，4604-token TTFT相对前一已推
  checkpoint `ea519e2e0f`的约2.924秒降到约2.608秒，累计提升约12.1%，达到
  checkpoint阈值。最终保留MFMA down split2/1040 blocks与attention 4 waves。
- TP-only MHC的192-CTA geometry原本只想服务单-token decode，但selector仅检查
  请求batch=1，导致单请求M1024 prefill也选择block_n=1。将其严格限制为
  `num_tokens==1`后，M1024 fused-tail micro从约`0.482 -> 0.342 ms`，三个输出
  逐元素bitwise exact；完整服务1028/4604-token约从`0.694/2.608 ->
  0.679/2.554 s`，再提升约2.2%。
- 8 MiB BF16 TP all-reduce实现对照：当前AIter peer-read约`0.679/2.554 s`；
  RCCL约`0.711/2.616 s`，SGLang固定顺序1-stage约`0.713/2.617 s`。后二者均
  退化，保留AIter；后续应减少/融合collective边界，而不是替换为RCCL。
- chunked-prefill严格ABBA：A=1024返回约`0.693/2.559 s`，B=2048约
  `0.545/2.433 s`（1028/4604 tokens），吞吐分别提升约27.2%/5.2%；4096约
  `0.546/2.436 s`，没有进一步收益。默认采用2048，既消除1028请求的4-token
  尾块，又把4604请求从5段减为3段。
- M2048 MFMA独立扫描中，gate/up split4/624 blocks约`7.50 ms`，优于沿用的
  split4/416约`7.74 ms`；down现有split2/1040约`6.42 ms`仍最优。因此只对
  M>=2048采用624-block gate grid，M1024继续416。CK仓库虽有FP4 microscaling
  MoE，但runtime正式支持列表不含gfx90a且MPerBlock=128；直接接入会重新引入
  小expert padding，暂不替换当前group32 CDNA2 MFMA路径。
- 最终默认2048+M2048 grid的1028-token六轮中位约`0.545 s`，4604-token五轮
  中位约`2.421 s`（HTTP TTFT口径约1902 input tok/s）。短256-token native AR
  为`74.53/74.95/74.87 tok/s`，3/3 hash仍为`51e2ac132057ead3`；4604+512
  两轮均完整`finish=length`，TTFT `2.413/2.422 s`、decode约46.67 tok/s。
- M2048新profile中TP0主要kernel sum为：routed gate/up `292.6 ms`、down
  `230.1 ms`、sparse attention `87.1 ms`、AIter peer AR `67.3 ms`、dense FP8
  `66.6 ms`、MHC stage0/post约`35.5/33.3 ms`、两次INT8 quant `24.8 ms`。
  因此即使prefill已接近2k，下一结构性目标仍是MoE数据复用/融合。
- CK-style gate N32原型让同CTA覆盖两个N16 tile；M2048保持split4时约
  `7.73 -> 7.70 ms`，仅约0.4%，split2更慢且改变归约。说明只放大N tile不能
  消除K循环/scale开销，原型已撤回；下一版必须显式LDS复用activation/scale或
  融合stage1→quant→stage2。
- 已有单CTA/token HIP `gfx90a_mhc_post_pre`在M2048下也不合适：完整融合约
  `1.248 ms`，当前两段Triton约`0.972 ms`，慢28%；其24x16384仍是标量FP16
  FMA。要替换MHC也必须写MFMA tile版，不能仅靠减少launch。
- Sparse paged-prefill attention继续扫描wave数，代表M2048、H16、D512、
  prefix512+extend128 micro为：8-wave `9.612 ms`、4-wave `4.872 ms`、2-wave
  `3.066 ms`、1-wave `1.958 ms`；四者输出逐元素bitwise exact。严格ABBA中
  A(4-wave)返回`0.545/2.421 s`，B(1-wave)约`0.526/2.280 s`（1028/4604），
  长prefill吞吐约2020 input tok/s，提升约3.6%/6.2%。B的短decode hash 3/3
  保持`51e2ac132057ead3`；4604+512两轮均finish=length，TTFT `2.269/2.278 s`、
  decode约46.10 tok/s。最终采用1 wave。
- `6680714cf8`之后增加纯HIP group32 INT8 quant专核：一个wave64用四个
  16-lane subgroup同时处理四组，每lane量化两个BF16元素。M2048真实shape
  micro中`[2048,4096]`约`0.380 -> 0.071 ms`，`[2048,6,512]`约
  `0.299 -> 0.062 ms`，量化值和scale均与原Triton逐元素bitwise exact；selector
  仅在gfx90a MFMA prefill且M>=1024时启用，decode不变。
- MFMA gate/down的E8M0 weight scale、activation scale和sort metadata广播改为
  wave shuffle后，M2048 micro约有gate 2--5%、down 2--3%局部收益。code-object
  审计发现共享assignment metadata使gate LDS从精确`16384 B`升到`16512 B`，
  跨过64KiB/CU的4-block residency边界；改成每wave寄存器加载再shuffle后，
  gate保持`104 VGPR/0 spill`并恢复`16384 B LDS`。单独端到端A/B约
  `2.224 vs 2.225 s`，没有可交付收益，但它改变了最佳grid档位。
- 真正的CDNA2 `mfma_i32_32x32x8i8` gate原型已用真实E256/M2048/topk6 shape
  验证：split4输出与16x16路径bitwise exact，但code object为`165 VGPR +
  32896 B LDS`，最好约`7.87 ms`，比当前16x16约`6.10 ms`慢29%；split2约
  `8.88--12.15 ms`。原因是16个C寄存器同时保留gate/up及scaled FP32累加后
  occupancy崩塌，原型已完全撤回。后续不应仅以减少MFMA指令数放大wave tile。
- 新16KiB LDS档位重新扫描M2048 MFMA grid：gate split4从624改1040 blocks，
  down split2从1040改624 blocks。平衡routing micro分别约`6.13 -> 6.05 ms`
  和`5.64 -> 5.28 ms`。实际TP4/EP1 no-A2A严格返回A：B为
  `2.182--2.190 s`，返回A为`2.225--2.230 s`（4604-token），稳定提升约1.8%；
  1028-token仍约0.516秒，因为不走完整M2048 tier。B的短decode三轮hash保持
  `51e2ac132057ead3`；4604+512两轮均`finish=length`、TTFT 2.188/2.189秒、
  decode 46.446/46.443 tok/s。长prompt completion hash仍会漂移，沿用既有
  teacher-forced要求，不把它当bitwise oracle。
- 针对DSV4 router的`M x 4096 @ 256 x 4096^T`正式扫描了本地CK BF16
  XDL/CShuffle实例。为ROCm 7.14/gfx90a只构建`gemm` profiler时，CK存在一个
  `DTYPES=bf16`下无条件声明F16别名并被`-Werror`拒绝的host-side小bug；将别名
  按`CK_ENABLE_FP16`条件化后可正常构建。120个实例的最佳时间（M=512/1024/
  1028/2048）约`86.7/90.0/90.1/98.6 us`，均慢于稳定torch/rocBLAS的约
  `34.6/43.7/43.4/62.5 us`，因此不引入CK runtime/实例扫描。
- 另写过固定N256/K4096、LDS协作搬运、wave64 BF16 MFMA的CK-style HIP专核，
  扫描64x64、128x64、128x128 block及K32/K64。输出相对torch约`2.4e-5--
  4.9e-5`，但最好仍仅约`0.43--0.52 ms`；简单同步LDS管线的K-tile barrier与
  bank/layout成本远高于rocBLAS，专核已撤回。后续若再做必须采用真正的异步
  双缓冲/CK blockwise copy，不能接入这个已证伪原型。
- AIter对上述gfx90a BF16 router shape没有tuned config，实际也是torch solution，
  但每次dispatcher fallback约`67--69 us`；直接`F.linear`约`35--63 us`。
  完整TP4/EP1 no-A2A服务严格ABBA（4604-token、每服务预热后7轮）为：A1
  中位`2.183 s`、B1 `2.186 s`、B2 `2.184 s`、返回A2 `2.184 s`。micro差异在
  完整prefill中完全不可见，故正式selector/env/script接线已撤回；不能把该
  fallback日志当成新的E2E瓶颈。
- 新增可复现的`scripts/bench_gfx90a_fp4_moe.py`，用真实TP4/EP1形状
  `E256/topk6/H4096/I512`和balanced expert排序元数据分别测gate/down；所有
  GPU实验前继续用`amd-smi process`确认无占用。rocprof硬件计数显示M2048旧
  32-assignment gate为`104 VGPR/16 KiB LDS/0 scratch`，每次约3.63亿VALU、
  1678万MFMA、1904万VMEM和2228万LDS指令，说明在线FP4解包/scale/地址计算
  明显重于MFMA本身。
- 尝试把packed FP4 routed权重离线展开并缓存为INT8，删除热循环内的`v_perm`
  解包；但gate从约`7.23 -> 10.45 ms`，因权重流量翻倍而退化44%，原型已撤回。
  这证明不能单独以显存换解包，后续若使用INT8缓存必须同时减少权重扫描次数。
- 将MFMA gate/down的expert sorter block从32参数化到64。M2048/topk6下每个
  expert平均约48条route，64-row CTA只扫描一次packed权重；gate的MFMA数保持
  1678万不变，VALU约`3.63亿 -> 2.56亿`、VMEM约`1904万 -> 1271万`，无scratch。
  资源变为`52 VGPR + 132 AGPR / 32 KiB LDS`。独立micro中gate约
  `7.26 -> 5.55 ms`（+23.6%），down约`6.01 -> 5.26 ms`（+12.5%），两阶段
  合计约+18.2%；32/64输出均逐元素bitwise exact。gate最佳使用split4/416
  blocks（416--1248基本同档），down使用split2/624 blocks；仅M>=2048启用，
  M1024及decode维持32-row路径。
- 完整TP4/EP1 no-A2A严格ABBA（4604-token，每服务预热后5轮）：A1旧32-row
  中位`2.184 s`，B1新64-row `2.061 s`，B2 `2.062 s`，返回A2 `2.185 s`，
  长prefill吞吐稳定提升约5.6%。两轮B短256-token native AR约
  `74.37/74.69 tok/s`且hash均为`51e2ac132057ead3`；新selector不触及decode。
  正式脚本通过`SGLANG_DSV4_GFX90A_FP4_MFMA64_PREFILL`默认启用该路径。

### TP8/EP1 首轮 bring-up 与 world-size-8 AR correctness（2026-08-26）

- 八个GCD均空闲且相互为一跳XGMI；GPU0--3属于NUMA0、GPU4--7属于NUMA1。
  已知node-0主存故障已由用户确认修复，因此8-rank服务使用
  `numactl --interleave=all`，不再把全部host staging强制绑到NUMA1。
- 为TP8/EP1接通raw packed-FP4 routed权重形状：
  `w13=(256,512,2048)`、`w2=(256,4096,128)`。8-rank加载后每GCD模型权重
  约26.80GB；`MAX_TOTAL_TOKENS=1048576`、`mem_fraction_static=0.96`下成功分配
  完整1,048,576-token pool，捕获BS1/2/4/8/16 graph后仍余约16.0GB/GCD。
- 默认AIter custom AR在TP8的8KiB BF16 reduction上自动使用8 CTA。BS1的
  256-token hash可稳定为`2629699770b9e036`、速度约65.5 tok/s，但相同prompt
  的BS2/4/8/16按batch slot分叉，故76.8/132.2/240.1/433.3 tok/s只记为无效
  诊断数据，不能作为性能checkpoint。
- direct-MoE关闭后的TP8 CKTile fallback约38 tok/s，单请求hash也漂移并产生重复
  文本，说明legacy CK权重重排/配置并未对TP8 shape完成correctness验证；该路径
  证伪。direct-MoE + RCCL则France固定IDs在BS1正确，BS2连续5轮共10请求全部
  逐token精确，因而把并发分叉限定到world-size-8 custom AR。
- SGLang legacy one-stage AR原先因DS环境缺少Python `amdsmi`而在cleanup调用未定义
  `amdsmi_shut_down`。已增加HIP peer-access fallback使其能初始化，但模型France
  oracle从BS1即完全错误，不能作为AIter替代。
- 将AIter `AITER_GFX90A_AR_SMALL_BLOCKS=4`作为唯一变量后，France固定输出在
  BS1为5/5、BS2为10/10逐token精确；256-token BS1六轮均为
  `2629699770b9e036`，稳态约`66.08--66.16 tok/s`。但更强的128-token同prompt
  跨slot检查推翻了短France结论：BS2仍可在generated token 3分叉，BS4/8/16也在
  3/24/82/112等位置分叉。把override扩到128KiB或降为1 CTA均不能修复；RCCL
  对照在BS2/4同样分叉。因此这不是AIter custom-AR/CTA竞态，而更接近TP8 attention
  或KV-slot相关的batch-shape数值差异。4-CTA不设为默认，相关阈值实验全部撤回。
  后续每个8-GCD性能改动必须先通过France IDs、BS1重复hash及至少128-token的所有
  捕获tier同prompt跨slot检查。
- 进一步关闭decode CUDA Graph、使用RCCL并在纯eager路径重复128-token同prompt
  检查：BS1自身稳定；BS2在generated token 3分叉；BS4分别在token 3/21分叉。
  因此CUDA Graph capture、AIter custom AR和Mori均已从该基础错误的必要条件中
  排除。当前应定位TP8下随batch/slot变化的attention、KV metadata或其他batch
  kernel；在逐token跨slot一致性修复前，TP8约65--67 tok/s只能标为correctness
  未通过的诊断性能，不能作为checkpoint。
- 8-GCD correctness gate固定为：官方France固定input IDs逐token精确；BS1重复
  hash；128-token同prompt在BS1/2/4/8/16所有目标tier逐token一致。任何代码、
  kernel或正式配置改动均先过该gate，再采纳性能数字；最终生成hash只用于发现
  分叉，定位根因时改用固定token/teacher-forced first-divergence。
- direct FP4 MoE存在一个真实但非首因的batch-shape数学分流：M=1在208个CTA内
  把activation量化进LDS，M>1则由runner先调用`per_token_group_quant_int8`。
  前者源自decode融合优化，后者源自prefill避免LDS随M扩张并复用一次量化，并非
  SGLang上游限制。两者的scale floor、除法顺序和round/clamp契约并不完全相同。
  将M=1也临时统一为外部quant后，随机duplicated-row oracle的gate/up和down在
  M1/M2之间均逐元素bitwise exact；但完整TP8 RCCL+eager仍在BS2 token 3、BS4
  token 3/21分叉，且BS2/4 hash集合与修改前一致，仅BS1轨迹改变。因此该改动已
  撤回且不作为修复提交。后续若要统一，应先让HIP group32 quant严格复刻Triton
  契约，再把同一helper融合回M1回收launch；当前首因仍在更早的batch-dependent
  hidden-state路径。
- TP8 RCCL+eager逐层/逐stage dump进一步定位：decode position 16的embedding、
  layer-0 MHC pre与attn norm在BS1/BS2间bitwise exact；首个差异出现在attention。
  layer-0相对L2依次为Q约`3.8e-6`、attention core约`2.8e-4`、wo_a约
  `8.6e-4`、wo_b/最终attn out约`4.4e-3`。关闭BS1专用fused inverse-RoPE
  不改变分叉，已证伪。主要放大源是cached-BF16 projection在M1走wave64 GEMV、
  M>1回退rocBLAS/einsum，使用不同归约树。
- 将普通wave64 BF16 GEMV和grouped wo_a GEMV扩展到M=1..16：每个token使用独立
  block网格与完全相同的wave64归约。三种真实普通projection shape及wo_a的
  duplicated-row micro在M1/2/4/8/16均bitwise exact。普通projection M2约比
  rocBLAS快40%，M4为持平到快34%，M8/16分别可能慢约1.1--1.8x/2.2--3.4x；
  grouped wo_a的M2/M4分别快约68%/48%，M8持平，M16慢约89%。整模型短gate中
  BS2两个slot由分叉变为逐token一致，但BS1/BS2仍可因attention core约2.8e-4
  数值差异在低margin token走不同greedy轨迹，BS4也可能分两类。此处应以
  duplicated-row bitwise、France固定IDs及teacher-forced logit tolerance作
  correctness证据，同时保留跨BS hash作为敏感诊断，不能把低margin greedy
  分叉单独解释为内存/竞态错误。端到端性能尚待graph+custom-AR ABBA后决定是否
  保留该batched projection改动。
- 最终按micro收紧selector：普通cached-BF16 projection只在M<=4使用batched
  wave64专核，grouped wo_a在M<=8使用；更大M回到能复用权重的rocBLAS/einsum。
  TP8/EP1 no-A2A、AIter custom AR、graph BS1/2/4/8/16下，France固定IDs在
  31/31请求逐token精确。256-token native AR每tier预热后3轮中位为
  `65.52/86.84/153.52/258.09/478.80 tok/s`（BS1/2/4/8/16）；相对首轮TP8
  诊断约`65.5/76.8/132.2/240.1/433.3`，BS2/4/8/16约提升13%/16%/7%/10%。
  这是当前可提交的8-GCD小batch projection checkpoint，但距离单请求120和多请求
  700仍远，后续需改变TP8 expert/collective分解而非继续只调projection。

### TP8/EP2 8-GCD结构对照与Mori恢复（2026-08-26）

- DS环境中的`amd_mori 1.2.3.dev56+g704e464a5`原为editable安装，指向已被清理的
  `/tmp/mori/python`。按同一commit `704e464a5`恢复源码；dry-run确认无依赖变更，
  使用`--no-build-isolation --no-deps`重建。gfx90a白名单、Debian上ROCm hsakmt
  错误硬编码`/usr/lib64/libc.so`的CMake remap，以及partial-EP subgroup broadcast
  必须使用`dist.get_global_rank(group,0)`的修复，已持久化为
  `scripts/rocm/patches/mori_gfx90a_partial_ep.patch`。后者对TP8/EP2的四个expert-TP
  lanes `[0,4]/[1,5]/[2,6]/[3,7]`是必要条件；固定`src=0`只会让第一组成功。
- TP8/EP2实际raw routed shape为`w13=(128,1024,2048)`、
  `w2=(128,4096,256)`，即128 local experts与expert-TP4。上游AIter CK两阶段
  FP4模板无论base Conda或系统ROCm 7.14 hipcc均在gfx90a报
  `Cannot select: llvm.amdgcn.raw.buffer.load.lds`，不能作为该shape回退。
  direct packed-FP4 shape白名单加入该精确组合后，Mori world-size2四组均成功初始化，
  graph BS1/2/4捕获完成。
- TP8/EP2 + Mori 16-block + direct FP4的France固定IDs在BS1/2/4共7/7逐token
  精确；256-token native AR三轮中位约`55.62/78.92/125.87 tok/s`。相比TP8/EP1
  的`65.52/86.84/153.52`三档均退化，因此TP8/EP2证伪为当前性能方向，不设默认；
  其价值仅为partial-EP correctness/bring-up oracle。
- 进一步尝试真正的异构分工：`TP8, DP attention=2, attn-TP4, EP2,
  moe-DP1`，形成attention groups `[0..3]/[4..7]`和跨组expert lanes
  `[0,4]/[1,5]/[2,6]/[3,7]`。服务可加载；graph模式也能捕获，但首请求后
  detokenizer心跳超时；关闭decode graph后France首请求仍90秒无返回。开启
  `--enable-dp-attention-local-control-broadcast`也不改变结果。日志仅显示收到请求的
  DP0进入prefill/MoE，DP1没有同步forward，导致Mori等待对端。因此当前SGLang
  scheduler不支持“单请求attention只在一个DP group、MoE却要求两个DP groups
  共同参与”的执行协议；需要新增remote/dummy MoE worker调度，不能靠现有flags实现。
  启动脚本已接入显式`DP_SIZE/ENABLE_DP_ATTENTION/
  ENABLE_DP_ATTENTION_LOCAL_CONTROL_BROADCAST/MOE_DP_SIZE`以便后续协议开发，
  但该组合不作为可用性能配置。

### 双TP4副本的8-GCD多请求checkpoint（2026-08-26）

- 8 GCD按`[0,1,2,3]`与`[4,5,6,7]`运行两个独立`TP4/EP1/no-A2A`
  native-AR副本。必须通过本脚本启动或完整继承其环境；手工只传server args会漏掉
  `unified_kv_triton`和gfx90a direct packed-FP4 MoE，前者在BS32退回TileLang DSA
  并触发`hipModuleLoadData`失败，后者会让BS32落入不支持AIter的通用FP8 runner。
- 每副本配置为`MAX_TOTAL_TOKENS=16384`、`SWA_FULL_TOKENS_RATIO=0.95`、
  `MEM_FRACTION_STATIC=0.90`，捕获`1/2/4/8/16/20/24/32`。graph max BS32要求
  `AITER_GFX90A_MXFP4_QUANT_MAX_ROWS>=192`；当前使用192。全部graph捕获约0.46GB，
  完成后仍余约12GB/GCD。
- 8K pool与SWA ratio 0.70只能同时admit约20个短请求/副本；24请求会分批prefill，
  因而即使存在BS32 graph也从约660 tok/s退到约350 tok/s。扩为16K pool并捕获精确
  BS20/24后消除了该scheduler/admission伪瓶颈。
- 两副本在每个捕获tier分别执行官方France固定input IDs，合计`214/214`请求均与
  9-token expected IDs逐token精确，每档输出唯一。256-token native AR、统一起跑、
  每档4轮结果：40并发trimmed约`889.92 tok/s`；48并发trimmed约
  `1005.39 tok/s`；64并发首轮冷态`641.38`，后三轮`1146.77--1158.73`，
  trimmed约`1152.69 tok/s`。所有请求均实际输出256 token。
- 因此8-GCD多请求`>=700 tok/s`目标已在严格native AR下通过；这属于副本并行，
  不改善单请求延迟。单请求`>=120 tok/s`仍未完成，后续不能用aggregate替代。

### DP-attention跨组A2A零token协议修复（2026-08-26）

- `TP8/DP-attention2/attn-TP4/EP2/MoE-DP1/Mori`原先能同步完成scheduler的
  MLP batch all-gather：DP0/DP1一致看到`global_tokens=[1,0]`并为DP1建立idle
  batch；因此此前挂死并不是controller漏发请求或idle scheduler未运行。
- 真正错误在DSV4 model层：A2A分支没有像no-A2A TP-MoE那样先做DP token
  gather。DP0进入shared/routed MoE时buffer为`[1,4096]`，DP1为`[0,4096]`，
  随后的full-TP shared/routed all-reduce以不同shape参与，导致设备端永久等待。
- 修复后，A2A+DP-attention在每层MoE前先replicate-gather hidden states与input IDs，
  在全局token buffer上按EP切source chunk并执行Mori，合并后再dp-scatter回attention
  owner。这样idle DP组参与MoE，但下一层不会重复执行另一个DP组的attention。
- graph capture需要禁用已知不安全的gfx90a Triton MHC+A2A组合；成功配置捕获全局
  BS4后完成post-warmup并健康就绪。France固定IDs连续5/5逐token精确，证明零token
  协议与scatter顺序正确。256-token native AR仅约`9.92--9.98 tok/s`，原因是每层
  新增DP gather/scatter、Mori和full-TP shared reduce的固定通信成本，故它是
  correctness修复而非性能checkpoint，不作为120 tok/s主路径。
- 同轮TP4/EP1单batch-overlap再次ABBA证伪：候选trimmed约`73.50 tok/s`，基线
  `74.79 tok/s`，两者hash同为`51e2ac132057ead3`；保持EP1默认关闭SBO。
- BF16-MFMA MHC K-split原型也证伪：当前FP16 split-K pre-mix约`83.5 us`，已有
  非split BF16-MFMA约`80.7 us`，新增split2/4/8/16均约`99--100 us`。原型未接
  selector且已完整撤回。
- 另测无Mori的`TP8/DP-attention2/attn-TP4/EP1/MoE-DP1`：France固定IDs
  连续8/8精确，但256-token native AR仅约`11.00 tok/s`。该布局的MAX_LEN
  decode把单请求按partial attn-TP对齐到M4；实验性保留真实M1/SUM_LEN没有改善，
  `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1`把scheduler metadata
  同步改到device group也没有改善（仍约`11.02 tok/s`）。两项实验改动均已撤回。
  这说明主要税来自通用DP-TP MoE每层full-TP gather/expert reduction/scatter及其
  graph协议，而非单纯dummy-row计算或CPU/Gloo控制面；BS1不应继续沿用通用DP路径，
  后续需要TP4主路径上的窄lane-pair远端worker协议。

### 复制attention-DP、full-TP8 MoE对照（2026-08-26）

- 为隔离通用DP gather/scatter固定税，实验性实现了latency-only replicated-DP：
  controller把同一请求复制给两个attention-DP组，各组独立运行attn-TP4并维护相同
  KV状态；DSV4 MoE与LM head跳过DP gather/scatter，八个rank直接以相同本地M进入
  TP8 collective；仅DP0向detokenizer发布输出。还必须同步绕过logits processor的
  DP gather/scatter及decode graph planner对global request-count metadata的要求。
- 该结构能捕获BS1/2/4 graph，并通过France固定IDs连续`5/5`逐token精确；256-token
  前两轮输出hash同为`50ad22ca8847feb3`。但端到端仅`21.744/21.737 tok/s`，显著低于
  TP4/EP1基线约`74.79 tok/s`。消除DP同步的收益远小于每层routed/shared expert及
  LM head从TP4扩到TP8带来的跨GCD collective固定税，因此该结构已证伪，所有原型
  代码均撤回，不保留实验开关。若继续做8-GCD单请求，必须维持latency-critical
  TP4主路径，只让额外GCD承担不扩大主collective的窄任务。

### `M>2` specialized-path设计扫描（2026-08-26）

- Inkling fused gate并不是`M>2`整体不可达。生产selector覆盖M1--4：M1/2使用
  register-sliced权重预载，M3/4仍走同一专核但改为shared-memory staging，M>4才
  回普通matrix multiply。阈值2来自VGPR/occupancy资源分桶；直接放宽寄存器路径会
  超出当前按worst-case wpt=4分配的寄存器数组，并可能spill，不是模型correctness
  限制。
- Inkling all-reduce的`M<=2`是另一回事：无exit barrier的v4协议依赖A/B双缓冲、
  reuse distance=2及全forward偶数次调用，属于协议安全边界；M>2切v5 push。它基于
  NVIDIA NVLS/multimem/PTX，不能移植为gfx90a XGMI优化，也不能擅自放宽。
- DSV4 gfx90a direct packed-FP4 MoE本身没有M>2禁令。M1在HIP kernel内量化一次，
  M>1走外部per-token group32 INT8预量化后仍进入sdot4专核；EP1/TP8等raw FP4 shard
  shape已在direct白名单内。真正的fallback浪费是通用AIter/CK按`block_m=32`处理
  极小M，而不是M>2 correctness。
- 可复用的最小实验是补齐DSV4 router的M1--4链路：BF16 projection GEMV底层已经
  支持M1--4，但后续native grouped Top-K严格锁定`[1,256]`，必须新写multi-row
  selector/kernel，不能只放宽TensorMatcher。目标是BS2/4，必须逐行对AIter oracle
  检查router logits、top-6 IDs及weights，再做France/teacher-forced与ABBA。它不会
  改善replicated-DP BS1，因为该布局每个attention-DP组的本地M仍为1。

### 8-GCD split-MoE单归约与slot-range专核（2026-08-26）

- `TP8 + attention-DP2 + attention-TP4 + MoE-DP2 + MoE-TP4`复制同一BS1请求；两组
  持有相同TP4 expert shards并计算互补Top-6 slots。初版3/3分工先做组内TP4 AR、
  再做rank-pair AR，France 5/5正确但仅约`70.26 tok/s`。
- 由于八个rank的local output正好是四个TP shard乘两组互补专家的partial，一次
  global TP8 all-reduce即可同时完成TP求和和DP合并。替换两级归约后France固定IDs
  5/5精确，默认AIter small-message geometry稳定约`72.57--73.74 tok/s`；显式
  `AITER_GFX90A_AR_SMALL_BLOCKS=1/2`分别仅约`69.66/71.40 tok/s`，均证伪。
- direct FP4 gate/down原固定208 blocks。3个live slots的真实shape扫描：52 blocks约
  `37.37/35.02 us`，104约`27.32/24.00 us`，156约`27.29/21.88 us`，208约
  `26.36/21.92 us`，260约`26.30/23.75 us`；各grid输出bitwise exact。减少grid
  并无端到端机会，208仍接近综合最优。
- shared expert只在DP0运行时，将routed分工改为连续2/4 slots后约`75.04 tok/s`；
  再把slot ownership下沉到gfx90a direct FP4 JIT specialization，gate只枚举owned
  tasks、down只量化/累加owned slots，同时保留上游`topk_ids=-1`语义mask，服务间
  稳态约`75.35--76.20 tok/s`。两段slot-range对旧mask实现逐元素bitwise exact，
  France固定IDs连续5/5正确，256-token hash在同一配置内稳定。
- 不能删除上游`topk_ids` mask：即使direct kernel partial对旧实现bitwise exact，
  服务级France会立即退化为单token EOS，说明runner外仍有消费者依赖sentinel；
  恢复mask后correctness恢复。以后不得用局部kernel oracle替代服务级逐token门禁。
- 两个结构对照均通过France 5/5但端到端退化，已撤回：attention heads从复制TP4
  改为真实TP8仅约`73.51 tok/s`，说明wo_b world8 AR税大于head计算减半收益；shared
  expert改为TP8并在两组都计算、routed恢复3/3仅约`72.32 tok/s`，原因是两组side
  stream shared都与routed争CU。当前保留方案仍是attention TP4复制、shared仅DP0
  TP4、routed连续2/4、单次global TP8 AR。
- 最后移除运行时已不再使用的MoE-DP pair与MoE-TP custom-AR communicator（attention
  TP4及global TP8仍保留AIter custom AR），新服务256-token稳态为
  `77.006/77.046/77.033/77.005 tok/s`，相对初版`70.26`约`+9.6%`。France固定IDs
  5/5精确；1024-token四轮均为同一hash `e39f2ee69e738527`，随KV增长的平均速度
  `59.107--59.179 tok/s`。该checkpoint仍远未达到120 tok/s，但已是当前8-GCD
  单请求路径的可信最佳值。

### M>2路径审计与router-linear融合反例（2026-08-26）

- 仓库不存在统一的“M>2不支持”。gfx90a native grouped router/Top-K严格只支持
  `scores=(1,256)`，真正已有M1--4能力的是它前面的BF16 router projection GEMV；
  M>1时projection仍可走wave64 HIP，但Top-K回到通用Triton。不能只放宽matcher，
  若扩展native router必须使用`grid.x=M`并让每个block处理一个token。
- Inkling gate中M1/2使用寄存器预载权重，M3/4改用shared-memory staging，是VGPR、
  occupancy与权重复用形成的性能分桶，不是correctness限制；该CUDA kernel不可直接
  搬到gfx90a，但“按M设计不同wave64 staging”可用于未来BS2/4专核。Inkling TP8
  all-reduce的M1/2边界则来自A/B双缓冲、reuse-distance=2及无exit barrier协议，
  又依赖CUDA NVLS/multimem，不能复用于ROCm通信协议。
- DSV4 direct FP4 MoE本身没有M>2禁令：M1在kernel内量化并通过shared memory复用，
  M>1由runner先做外部group-32 INT8 quant；当前split-MoE BS1的每组本地M仍为1，
  所以迁移Inkling M3/4策略不会改善当前单请求。MHC的`global_batch_size==1`限制
  主要用于保证各rank graph捕获相同kernel/collective序列，扩到BS2/4必须另做
  rank-invariant验证。
- 实验性HIP kernel将learned-router的BF16 `[1,4096]x[256,4096]` projection、
  sqrt-softplus、bias、Top-6 renorm与2/4 slot mask融合为一次launch。32 CTAs各算
  8个logits，最后CTA通过atomic ticket完成Top-K。随机micro中logits/IDs一致，weights
  仅有`1.49e-8--5.96e-8`差异；standalone约`16.9--17.1 us`，看似远快于分离调用。
  但完整graph开启态七轮稳定约`73.47--73.54 tok/s`，关闭态热稳态约
  `76.13--76.19 tok/s`，端到端反而退化约3.5%。32-CTA驻留、threadfence/atomic
  以及图内调度成本抵消了省下的launch，原型与开关已完整撤回。
- A/B服务均偶发`/generate` 502，而随后的256-token benchmark连续成功且hash各自
  稳定；该现象不随融合开关出现，需作为DP local-control/output-suppression的独立
  服务包装问题倒查，不能归因于router数值。

### 8-GCD split路径的共同轨迹latency oracle（2026-08-26）

- 当前真实路径约`76.1--77.0 tok/s`，即约`13.0 ms/token`。以下oracle均只为
  latency decomposition，刻意破坏模型语义，实验后代码和开关全部撤回；不能作为
  correctness或可交付吞吐。
- 单独删除shared expert约`78.67--78.70 tok/s`，相对同轮基线只减少约
  `0.3 ms/token`；shared并非当前第一瓶颈。删除router+routed FP4 MoE但保留shared
  和TP8 layer-end AR约`100.36--100.48 tok/s`；删除全部MoE及该TP8 AR约
  `109.47--109.61 tok/s`。即使MoE完全免费，现有其余路径仍达不到120。
- 为消除不同输出导致expert weight-cache轨迹不同的干扰，attention分解全部在
  self-attention结果清零、completion hash固定为`41d90b79a4feb5d6`的共同轨迹上
  对照：完整执行attention约`13.37 ms/token`；整个attention跳过约`6.61`；只跳
  paged core约`12.18`；保留prepare+core但跳inverse-RoPE/wo_a/wo_b/TP4 AR约
  `11.39`。解得prepare约`3.59 ms/token`、paged core约`1.19`、output half约
  `1.98`，attention合计约`6.76 ms/token`。
- 在相同清零轨迹中同时跳过core/indexer两条compressor更新约
  `13.27 ms/token`，只比完整attention少约`0.10 ms/token`；短上下文下compressor
  已被multi-stream隐藏，prepare大头是Q/K/V projection、norm/RoPE/cache pipeline。
- 现有wave64 projection真实shape为`wqkv_a=[1536,4096]`、
  `wq_b=[8192,1024]`；批量event micro均约`15--16 us`/launch，已显著快于torch
  的约`28--33 us`。旧INT8-weight kernel在当前代码反而约`16.5--16.9 us`，不再
  快于BF16。一次全局activation quant再由dot4 kernel复用的两-launch原型与旧INT8
  bitwise exact，但约`26.9--27.5 us`，额外launch和global input读取使其证伪并撤回。
- 现有single-CTA native sqrt-softplus Top-K在当前split路径France 10/10精确、
  hash稳定，但仅约`73.34--73.43 tok/s`，与atomic fused-router的约73.5相同，均
  低于generic Triton约76--77；说明低CTA router在整图调度中不如Triton并行供给。
- ROCm Kineto在stop trace时崩于`libkineto::RocprofActivityApi`；rocprofv3 attach
  又受multiprocessing attach注册边界限制。包裹进程的延迟采集会污染`hipconfig`/
  `rocminfo` stdout，并在采集窗口结束时令HIP graph capture失步。临时解析补丁和
  profiler进程均已撤回；不得在该路径再次用在线Kineto/rocprof包裹graph服务。

### TP4 decode HIP/FP16局部探针（2026-08-26）

- direct FP4的四nibble解包原本用selector构造加两次CDNA2 `v_perm_b32`。实验性
  256-entry byte LUT用两个constant-memory读取一次解两个nibble；真实TP4 shape
  micro反而将gate从`28.36 -> 45.05 us`、down从`23.39 -> 30.50 us`。随机lane
  索引不能形成便宜scalar broadcast，现有byte permute明显更合适；原型完整撤回。
- M1 direct kernel当前由一个线程串行量化一个group32，但128/96个group线程已并行
  覆盖gate/down。改成16-lane subgroup在同一launch内协作量化后，输出对scalar
  oracle逐元素bitwise exact；ABBA micro却让gate约`28.54--28.65 ->
  29.76--29.83 us`，down约`23.52--23.70 us`持平。shuffle/协作成本没有回报，
  原型完整撤回。外置HIP quant连同额外launch则约`36.14/36.63 us`，更慢。
- 仓库已有未接线的BF16-input/FP16-weight HIP wave64 GEMV。真实projection micro中
  `N256,K4096`约`13.46--13.62 -> 7.67--7.73 us`，FP16结果相对原FP32权重的最大
  误差还略小于BF16；但N1536仅快约7%，N4096/N8192约1--2%，说明大shape已受带宽
  限制。将FP16只接到40个learned-router层后，完整TP4/EP1/no-A2A、graph BS1
  ABBA为A1约`73.49`、B1约`73.70`、B2约`73.53`、A2约`73.84 tok/s`，没有可复现
  收益；所有服务France固定IDs正确，A/B各自长hash稳定但不同。接线与额外缓存均
  撤回，保留原仓库HIP module，不把单projection micro收益冒充端到端收益。
- 结论：当前约`13.6 ms/token`不能再靠5--6us/layer的小projection优化达到
  `8.33 ms/token`。下一结构实验应维持两个attention TP4组，给两组复制TP4 expert
  shard并按top-6 slots分工，只交换一次4096维rank-local partial；绝不能让MoE或
  LM head重新扩大到full TP8 collective。

### CDNA2 BF16 MFMA用于M1 GEMV的反例（2026-08-26）

- CK对gfx90a暴露`64x4x4 BF16 -> FP32` MFMA，可把单token GEMV映射成64个weight
  rows乘一个live activation column，另外三列补零。实测lane mapping正确：在
  `N=1536,K=4096`随机输入上，MFMA输出转BF16后与现有wave64 VALU专核逐元素一致。
- 但单wave需要沿K串行发出1024条MFMA，约`193--213 us`；改成2/4/8 waves对同一
  64 rows做split-K后分别约`132.5/97.2/95.7 us`，仍远慢于现有专核约
  `13.46--13.60 us`。根因是M1映射浪费3/4矩阵列，且每个K4仍有长依赖链和额外
  partial reduction；不是简单接入CK就能获得收益。
- 原型、wrapper和selector均完整撤回。MFMA应保留给M足够大、至少能填满4列且能
  沿M/N复用权重的BS4+路径；BS1 projection继续使用当前wave64向量读取+VALU归约。

### q_lora RMSNorm backend ABBA（2026-08-26）

- `[1,1024]` BF16 standalone直接调用时，AIter RMSNorm约`10.01 us`，仓库现有
  HIP-JIT RMSNorm约`4.45 us`；两者最大BF16输出差`0.03125`。经`RMSNorm`模块在
  eager Python循环测得的约80us主要是host enqueue，不代表graph中的kernel时间。
- 只将43层q_lora切到JIT后，France固定IDs 5/5仍精确；A长输出hash
  `14593da264d38f29`、B为`bf80c4a9b3acaecd`，各自稳定但因reduction顺序不同而分叉。
  ABBA受到跨服务频率漂移：A1约`76.96`、B1/B2约`76.55--76.57`、A2约
  `75.75--75.86 tok/s`。两端A平均后差异仅约0.25%，没有可复现端到端收益。
- selector和接线完整撤回。后续不得用未捕获的Python module循环延迟直接外推
  CUDA graph；此类小norm替换必须服务级ABBA过门槛才保留。

### unified-KV Q/K norm+RoPE HIP融合反例（2026-08-26）

- 生产Triton kernel在一个launch内完成16个local Q heads的RMSNorm+RoPE，以及K
  RMSNorm+RoPE+BF16 ring store。独立真实shape约`43--55 us`。两段现有HIP专核加
  新BF16 unified-store K专核可到约`18.99 us`，但服务只有约`74.5 tok/s`。
- 将Q的2个workgroups和K的1个workgroup进一步融合为单HIP launch，standalone约
  `8.05--10.06 us`；使用与Triton相同的BF16 cos/sin后K cache逐元素bitwise一致，
  Q因RMS reduction顺序仍有最大`0.03125`、平均约`1.61e-4`差异。France固定IDs
  10/10精确，长输出hash在配置内稳定。
- 端到端仍只有约`74.7--75.1 tok/s`，关闭multi-stream后约`74.7 tok/s`，均低于
  同期A约`75.8--77.0 tok/s`。说明standalone launch latency不是该graph的充分
  指标；Q轨迹改变及整图调度抵消了micro收益。两核、单核、selector全部撤回。
- 原Triton prologue的`num_warps=1/2/4`输出逐元素bitwise一致，standalone分别约
  `44.00/43.75/43.29 us`，当前4 warps已最优；8 warps约`43.10 us`但K输出明显
  损坏。临时wrapper参数化已撤回，不应再扫这一维。

### TP4/TP8 small-message AIter AR分离扫描（2026-08-26）

- 8KiB BF16消息的AIter one-stage默认grid随world size变化：TP4为4 CTAs、TP8为
  8 CTAs。旧`AITER_GFX90A_AR_SMALL_BLOCKS=1/2`同时压低两者，不能单独判断
  attention TP4 AR。临时增加按world-size覆盖后，TP4 graph micro按最慢rank计，
  1/2/3/4 CTAs约`14.39/12.95/9.09/9.62 us`，3 CTAs略优于默认4。
- 完整8-GCD split服务只设TP4=3、TP8维持默认8，France固定IDs 10/10精确，256-token
  hash仍为基线`14593da264d38f29`；热稳态仅`75.21--75.26 tok/s`，低于同期基线
  `75.75--75.86`。micro优势被完整graph中的CTA/barrier调度抵消。
- world-size覆盖和测试脚本参数均已撤回；AIter外部仓库恢复到实验前状态。不要再用
  全局small-block变量推断某一个collective group的收益。

### `M>2` 分支语义与可复用边界（2026-08-26）

- 仓库不存在统一的“`M>2`不支持”。gfx90a native grouped router/Top-K才是严格的
  `M=1`专核：Python selector和HIP `TensorMatcher`均固定`[1,256]`，kernel也没有
  token/block维度。此前所谓已有`M=1..4`能力指的是router前面的BF16 projection
  GEMV，不是Top-K本身；M>1会回退通用Triton router。
- Inkling gate中的`M<=2`是CUDA资源分桶：M1/2把权重切片驻留寄存器，M3/4改用
  shared-memory staging并由多个warp分摊；这是VGPR、occupancy和权重重用的性能
  选择，不是correctness限制。可复用的是“按M选择寄存器/shared staging”的设计，
  不能直接复用其CUDA kernel。
- Inkling TP8 all-reduce的M1/2 v4路径则是通信协议安全边界：它省略exit barrier，
  依赖A/B双缓冲、reuse distance=2及每次forward偶数次AR。该实现依赖NVLS/
  multimem，既不能简单放宽到M>2，也不能直接移植到ROCm。
- AIter gfx90a custom all-reduce没有M2上限；8KiB小消息的主要问题是每层固定barrier
  与CTA驻留。DSV4 direct FP4 MoE同样支持M>1：M1在kernel内量化，M>1先做外部
  group-32 INT8量化，避免按expert/CTA重复量化。
- 对当前BS1，两个DP副本的本地M仍为1，因此照搬Inkling M3/4 staging不会改善单请求。
  值得复用的后续工作是：为native grouped router增加`grid.x=M`的一-token-per-block
  M1--4实现并测BS2/4；以及在真实M1/2/4 MoE shape上比较核内与外置量化。任何扩展
  都必须逐行核对router top-k IDs/weights、MoE输出和graph各rank collective顺序。

### TP4 attention AR + MHC post融合反例（2026-08-26）

- AIter已有TP4/gfx90a专用`fused_mhc_post`：用一个`4 CTA x 512` peer-read kernel
  同时完成4096维BF16 attention all-reduce、四通道MHC post以及64个RMS partial。
  临时接到8-GCD的`TP8/DP2/attention-TP4` decode路径，并将partial直接交给现有
  split-K MHC pre-mix；graph BS1八rank均能稳定捕获，没有设备端自旋。
- correctness方面，France固定IDs连续10/10逐token精确；融合配置的256-token hash
  连续7轮稳定为`bee97dfcc2868264`。它与基线`14593da264d38f29`不同，来自peer-read
  reduction顺序变化，但没有观察到配置内漂移。
- 服务级ABBA：A1热稳态约`75.52 tok/s`，B约`75.42 tok/s`，返回A2约
  `75.95--75.99 tok/s`（忽略首轮频率/JIT样本）。融合约回退0.7%，没有收益。
  固定四个512-thread CTA的MHC四通道计算、RMS归约和barrier驻留抵消了少一个launch
  的收益；接线、selector和环境开关已完整撤回。除非先重做更轻的CTA/wave几何，
  不应再次把该现成AIter接口直接接入生产路径。

### bitwise-exact HIP Q/K norm+RoPE+BF16 cache probe（2026-08-26）

- 根据生产Triton LLVM IR复刻了gfx90a的四wave RMS归约顺序：wave内使用
  DPP `row_shr 8/4/2/1`、`row_bcast 15/31`，跨wave按
  `(wave0+wave2)+(wave1+wave3)`累加。HIP专核将16个Q head、K RMSNorm、RoPE及
  unified BF16 cache store合并为一次launch；standalone可由约`84.8 us`降至
  `21.4--21.8 us`，且Q输出、原位K和cache row均逐元素bitwise一致。
- 完整TP8/DP2 graph扫描每CTA处理`1/2/4`个Q heads。France固定IDs各10/10精确，
  256-token hash始终为基线`14593da264d38f29`。hpb1热稳态约`76.09--76.14 tok/s`，
  hpb2约`76.45--76.56 tok/s`，hpb4约`76.41--76.49 tok/s`；同期A约
  `75.86--75.97 tok/s`，最好收益不足1%。
- 结论：standalone launch latency大幅下降并不代表完整graph收益；CTA驻留和其它
  streams/collectives的调度抵消了该micro收益。selector、环境开关和HIP专核均已
  撤回。后续除非能与attention core或下游collective进一步融合，不应再次单独替换
  Q/K prologue。

### learned-router + direct FP4 gate/up融合反例（2026-08-26）

- direct FP4的M1核内group-32 activation quant已被weight streaming隐藏：真实接口
  micro中gate/down传入预量化buffer与核内量化几乎同速，而独立量化launch约增加
  `60 us`；两条量化契约还存在scale/rounding差异。因此未采用“把量化前移到MHC”方案。
- 原型把40个learned-router层的BF16 `[1,4096]x[256,4096]` GEMV、
  sqrt-softplus Top-6和direct FP4 gate/up合并到一个kernel。前32个CTA计算router，
  最后到达CTA完成Top-K，epoch/seen协议允许graph反复replay；前三个hash-router层保持
  原路径。随机oracle中router logits、Top-6 weights、owned IDs和gate/up输出均与
  分离native链bitwise exact，连续300次208-CTA replay无死锁。
- standalone ABBA中104-CTA融合约`76 us`，分离链约`91--97 us`，表面可节省
  `15--21 us/learned layer`。完整TP8/DP2 graph却回退：104-CTA约`72.50 tok/s`，
  208-CTA约`73.13 tok/s`，同期A约`76.60--76.63 tok/s`。France固定IDs分别
  10/10与5/5精确；B的256-token hash稳定为`434f12104b0d1f9d`，A为
  `14593da264d38f29`，差异来自native Top-K reduction轨迹。
- 结论：graph中的epoch等待、额外LDS/VGPR footprint及CTA调度抵消了省下的launch；
  kernel、handoff、selector和环境开关均完整撤回。除非改成无全grid等待的producer-
  consumer或硬件cooperative launch，不应再次使用此融合协议。

### split-MoE shared overlap + 3/3 routed slots反例（2026-08-26）

- 当前可信布局在MoE-DP0串行计算shared expert+前2个routed slots，DP1计算后4个
  routed slots。实验将DP0 shared放到辅助stream，并把routed ownership改为连续3/3，
  期望用并行shared换取最慢routed分支由4 slots降为3。
- TP8/DP2 graph BS1八rank均能捕获，France固定IDs 10/10精确，256-token hash在配置
  内稳定为`9e98303cfefcd3ce`。但热稳态仅约`75.03--75.29 tok/s`，低于同期串行2/4
  基线约`76.60--76.63 tok/s`。
- 结论：shared与routed的CU/缓存竞争超过3/3分工带来的收益。辅助stream接线、slot
  参数和环境开关均撤回；当前2/4串行并非遗漏开关，而是更好的端到端平衡点。

### 8-GCD decode clock residency检查（2026-08-26）

- `PERF_LEVEL_AUTO`空闲时为800 MHz，但长native-AR请求期间八个GCD连续采样均为
  `100% GFX`、`1700 MHz`，无throttle；四张卡有功耗读数的主GCD约`340--356 W`。
- 因此锁`PERF_LEVEL_HIGH`不会解释当前约76到120 tok/s的缺口；不用再把跨服务的
  小幅频率漂移误判为主要性能机会。当前限制仍是graph内计算/collective结构。

### attention wo_a→wo_b persistent融合反例（2026-08-26）

- HIP原型用常驻grid在一个launch内依次完成本地`wo_a [2,4096]→[2,1024]`和
  `wo_b [2048]→[4096]`，中间仍写BF16，wave64 dot归约顺序与现有两kernel一致。
  104/208 CTA均在随机输入上实现中间值和最终值逐元素bitwise exact；连续300次
  epoch barrier replay无死锁。
- standalone ABBA中104 CTA约`50--53 us`，208 CTA约`48.8--49.2 us`，分离链约
  `55.7--65.1 us`。但完整TP8/DP2 graph中104 CTA仅约`75.72--75.78 tok/s`，
  208 CTA进一步降至约`73.95--74.01 tok/s`，同期A约`76.60--76.63 tok/s`。
- France固定IDs在104 CTA为10/10、208 CTA为5/5精确，256-token hash均保持基线
  `14593da264d38f29`。退化纯属persistent grid barrier/CTA驻留改变graph调度；原型、
  selector和环境开关全部撤回。后续若融合attention output，必须避免全grid barrier，
  或直接并入已有collective/MHC消费者而不是用新的常驻同步核。

### TP8 MoE AR + MHC post融合与AIter数值修复（2026-08-26）

- 将AIter现有TP4 `fused_mhc_post`泛化为TP8：每rank一个CTA，TP8时每CTA两个
  32-lane subgroup，八个CTA仍输出固定16个RMS partial/channel。独立8-rank graph
  micro的分离`TP8 AR + Triton MHC post/RMS`最慢rank中位约`26--28 us`，融合约
  `23.7--25.0 us`，micro看似快约9--13%。
- 独立CPU/Gloo oracle暴露了原TP4融合核已有的真实correctness bug：MHC定义为
  `out[j] += comb[k,j] * residual[k]`，kernel却读取`comb[j,k]`，把mixing matrix
  转置。修为`comb[input_channel * 4 + channel]`后，随机BF16融合输出对Triton oracle
  逐元素完全一致；RMS partial最大差仅`4.88e-4`，来自局部求和顺序。
- AIter eager wrapper还把uint8注册buffer直接传给要求BF16的C++入口，必然触发dtype
  检查。将所需byte slice先`view(input.dtype).view_as(input)`后，eager与graph随机
  oracle均通过。这两项属于数值/API修复，不应与性能收益混为一谈。
- 临时将split-MoE层尾TP8 AR延迟到下一层MHC post：BS1 graph八rank稳定捕获，France
  固定IDs连续10/10精确，256-token hash连续7轮稳定为`2afddbaf14d77f25`。但服务级
  ABBA中融合约`75.84--76.10 tok/s`，返回基线约`76.93--77.00 tok/s`，端到端回退
  约1.2%。生产接线及环境开关已完整撤回；除非重做更轻的CTA/尾部协同，不再启用。

### prequant INT8 q-lora/q_b融合反例（2026-08-26）

- 先补扫attention output现有wave64几何：`kRows=2,waves=4`的grouped `wo_a`+
  `wo_b`约`33.89--33.94 us`，慢于当前`kRows=1,waves=4`的约`32.76--32.83 us`；
  `kRows=4,waves=4`进一步退到约`36.55--36.57 us`。三者输出均逐元素bitwise一致，
  当前几何仍是较优点，不再接服务。
- 给per-row INT8 weight GEMV做过仅用于lower-bound的prequant入口。三个真实shape
  `N,K=(1536,4096)/(8192,1024)/(4096,2048)`在预分配output下，BF16 wave64约
  `12.9/15.9/15.9 us`，纯prequant INT8 scan约`5.5/6.7/7.2 us`；prequant与原
  kernel内部per-tensor量化输出逐元素bitwise一致。说明INT8 weight scan本身有潜力，
  但独立wrapper/额外量化不能代表graph收益。
- HIP单CTA原型将M1 `[1,1024]` q-lora RMSNorm同时输出BF16、per-tensor INT8与scale。
  16 waves/1024 threads约`4.43 us`，对FP32/Torch RMSNorm及其自身量化oracle均逐元素
  bitwise一致；与prequant q_b合并的standalone链约`11.1 us`，相对AIter RMSNorm+
  BF16 q_b的约`19.3 us`表面节省约`8 us/layer`。
- 完整TP8/DP2 graph中同时启用既有三个INT8 projection cache及该q_b融合：France
  固定IDs 10/10精确，256-token 8/8 hash稳定为`777c3b8757e7edae`；关闭态hash为
  `14593da264d38f29`。B热稳态约`75.58--75.71 tok/s`，同期A约
  `75.49--75.51 tok/s`，收益不足0.3%，远低于采用门槛。
- 结论：projection kernel的局部节省仍被compressor/多流graph调度隐藏。prequant
  接口、RMS+INT8 HIP核、selector及环境开关全部撤回；若以后重做weight-only路径，
  必须让同一份量化activation跨wqkv/router/MoE消费者复用并改变临界依赖，而不是只
  替换单个q_b producer-consumer链。

### 8-GCD host NUMA启动固定（2026-08-26）

- 同一TP8/DP2代码未加host NUMA策略时热稳态约`75.49--75.51 tok/s`；显式
  `numactl --interleave=all`的首个服务约`77.06--77.08 tok/s`。这是host staging、
  scheduler及JIT内存落点造成的系统差异，不是GPU kernel收益。
- harness现在对`TP_SIZE=8`默认加`numactl --interleave=all`，较小部署默认不变，
  可用`NUMA_INTERLEAVE_ALL=0|1`覆盖。自动路径的`/proc/<pid>/numa_maps`有2944个
  `interleave:0-1`映射，主进程node0/node1内存约`639/649 MB`，确认不是注释性开关。
- 自动路径France固定IDs 5/5精确，256-token 4/4 hash保持
  `14593da264d38f29`；第二个服务热稳态约`76.36--76.40 tok/s`，说明仍有服务间
  约0.7 tok/s漂移，但已消除遗漏外层NUMA wrapper造成的系统性回退。后续所有8-GCD
  A/B必须通过同一harness启动，不再手写不同的numactl前缀。

### CDNA2 v_dot2 MHC split-K stage0反例（2026-08-26）

- 当前M1 fused MHC先用`24 rows * 8 splits = 192`个Triton单-wave CTA生成FP16-weight
  dot partial，再由单CTA fused tail完成partial/RMS reduction、Sinkhorn、weighted
  residual与RMSNorm。HIP原型让每个`(row,split)` CTA直接用CDNA2
  `v_dot2_f32_f16`，不增加grid barrier。
- 预分配output的stage0 micro中，Triton约`20.2--20.7 us`；HIP每task 1/2/4 waves
  分别约`3.6/3.26/3.10 us`。192个partial对FP32 oracle cosine约`0.99999988`，
  最大差约`3.6e-5`。完整函数wrapper表面从约`107`降到`90 us`，一度看似可提供
  超过5%的整模型收益。
- 服务结果推翻该外推：4-wave HIP France 10/10精确、256-token hash稳定为
  `1d5c45e0caac9601`，热稳态约`76.44--76.55 tok/s`，同期A约`76.36--76.40`；
  1-wave France 5/5精确、hash稳定为`30e9e330b5b7affe`，却只有
  `75.43--75.55 tok/s`。两者均未形成可采用收益。
- 更严格的预分配分解显示：Triton stage0约`19.7--21.2 us`、stage1约`15.9--16.0
  us`、仓库已有HIP finish仅约`7.26--7.36 us`。此前`~105 us`完整wrapper计时混入
  动态output allocation；graph replay的真实kernel链远短于此。stage0又与其它stream
  重叠，因此6倍局部加速不在critical path。
- HIP stage0、selector和环境开关已完整撤回。今后MHC micro必须预分配所有output并
  分别计stage0/stage1/finish；不得再用Python wrapper链时间推断CUDA graph收益。

### HIP graph内torch Event阶段计时不可用（2026-08-26）

- 临时在layer20的attention-pre、attention、FFN-pre、MoE边界记录
  `torch.cuda.Event(enable_timing=True)`，并在decode graph replay后同步读取。即使
  关闭dual sparse graph、只捕获dense graph，仍得到单层约2.7ms、明显超过整token
  约13ms/43层的矛盾结果。
- 原因是当前HIP/PyTorch栈没有让这些Python Event对象在graph replay时形成可读取的
  新时间戳；结果实际来自capture/eager warmup的M256轨迹。France固定IDs 5/5仍精确，
  但该计时不能作为性能证据。探针已撤回；不要再次用capture内torch Event拆阶段。

### AIter graph peer metadata与shared-add归约融合反例（2026-08-26）

- 临时给AIter `CustomAllreduce`增加只读metadata入口，返回已注册输入对应的device
  `RankData`、本地/peer signal地址、rank与world size。8-rank oracle证明：复用AIter
  自身预注册buffer的eager路径，以及capture后执行base+offset注册的普通allocator
  graph输入，都能被SGLang HIP JIT正确peer-read；4096个BF16元素的TP8求和逐元素精确。
- oracle同时暴露AIter eager `register_buffer()`的现有陷阱：对小PyTorch caching-
  allocator子分配，`_get_ipc_meta()`只传播HIP IPC handle并把offset固定为0，remote
  rank会读到同一allocation slab的错误地址。graph注册会通过
  `hipPointerGetAttribute`计算真实base+offset，因此当前正式decode graph不受影响；
  以后不得用eager注册的小子分配验证peer kernel，除非先修该offset协议。
- 独立graph ABBA中，现有AIter 8KiB TP8 AR中位`16.441 us`；使用保留signal槽的
  SGLang 8-block peer-reduce为`18.510 us`，慢约12.6%。再把DP0四rank的BF16
  `routed += shared`折进entry barrier前：分离add+AIter为`17.900 us`，融合实现
  `18.450 us`，仍慢约3.1%。因此它不具备接服务或并入direct-down的收益门槛。
- 两套JIT均未可靠地因被include header变化自动失效：AIter模块和SGLang实验模块都
  曾复用旧`.so`，必须移动缓存后才执行新签名。该方向的metadata接口、JIT kernel、
  benchmark和生成模块均已撤回/恢复；若以后做producer-row overlap，需先修cache key，
  并直接复用AIter更高效的8-warp cooperative load，而不是每线程串行读八个peer。

### gfx90a graph-replay realtime marker与attention prepare分解（2026-08-26）

- HIP graph capture内的`torch.cuda.Event`不能提供可信replay时间戳后，新增默认关闭的
  gfx90a realtime marker：单线程HIP kernel读取`__builtin_amdgcn_s_memrealtime()`
  并写入预分配`uint64` tensor。独立capture/replay校准约`40 ns/tick`，能在graph每次
  replay中更新；仅当`SGLANG_DSV4_GFX90A_REALTIME_TRACE_LAYER`与
  `SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY`显式设置时启用。
- TP8/DP2、EP1/no-A2A、graph BS1的layer20粗分解，按33组完整replay逐组取8 rank最慢
  值：attention prepare约`85 us/layer`、attention core约`25--28 us/layer`、output/
  inverse-RoPE+projection+TP4 AR约`41--50 us/layer`、MoE+TP8 AR约`90--100 us/layer`。
  单层总span约`289 us`，乘43层与约`13 ms/token`的客户端稳态一致。
- layer20为c4层。细分prepare同样按33组replay的max-rank中位数：`wq_a 14.88 us`、
  q-norm `6.88 us`、`wq_b 16.80 us`、fused qk-norm/RoPE/cache store `7.52 us`、
  indexer `18.08 us`、compressor `21.12 us`；marker/return固定开销约`4.5 us`。所以此前
  合并观察到的约`37 us`尾段不是一个巨核，而是indexer与compressor两条固定成本链。
- 三次诊断服务均通过France固定IDs 5/5；最终服务256-token 4/4 hash为基线
  `14593da264d38f29`，热稳态`76.18--76.27 tok/s`。中途marker误接到未启用的HIP
  multi-stream与相邻NPU路径时，槽位保持0；对应负数/超大差值明确判无效，不作为性能
  证据。后续诊断logger应对零槽与非单调时间戳fail-loud。

### TP8 split-MoE细分、并发与active-grid反例（2026-08-26）

- AMD原有`_forward_prepare_multi_stream_hip()`在后续NPU refactor中因MQALayer
  selector只保留CUDA/NPU而变为不可达。临时恢复HIP selector后graph能捕获、France
  10/10及256-token hash均保持基线，但热稳态仍约`76.13--76.17 tok/s`，没有超过
  串行路径。真实projection graph micro进一步解释：两条512 GEMV串行约`14.01 us`、
  side-stream并发约`22.22 us`；生产512+2048组合串行约`25.66 us`、并发约
  `34.61 us`。跨stream fork/join及CU/HBM竞争大于重叠收益，selector恢复已撤回。
- 为测试CK-style compressor专核，实现过单个512-thread HIP CTA融合paged state更新、
  online softmax pool、RMSNorm、RoPE，并进一步融合c4 FP8 quant/paged store。c4/c128
  数值oracle的state逐元素一致，输出max error约`4.8e-7/1.9e-6`；直接store的scale
  byte完全一致、576-byte value区仅24 byte不同。但预分配graph ABBA中，c4现有两段
  Triton为`8.72 us`、HIP为`9.78 us`；c128为`12.30/60.28 us`。连store后现有三段
  `10.18 us`、HIP单核`10.97 us`。单CTA丢失跨CTA维度/slot并行，原型完整撤回。
- split-MoE当前是DP0/DP1按Top-6的`2/4` slots分工。direct FP4默认208 blocks；真实
  TP8 shape micro显示active-grid不是首因：2-slot gate blocks64/104/208约
  `18.21/18.22/18.28 us`，4-slot gate 104/128/156/208约
  `30.83/29.52/28.93/29.59 us`；down最佳仍约156--208 blocks。所有输出逐元素一致。
- layer20 marker显示DP0 MoE span约91--97us、DP1约97--103us。3/3 slot实验France
  10/10、hash稳定为`84208f7692fab60b`，约`76.40--76.48 tok/s`；同期2/4为
  `75.82--76.01 tok/s`，仅约0.6--0.8%，低于采用门槛。2/4的用途是用DP0 shared
  expert补偿DP1更多routed slots，阈值实验接线已撤回。
- realtime trace扩展到MLP内部后，33组replay逐组取8 rank最慢值：shared-pre
  `20.64 us`、router `7.52 us`、Top-K+mask `18.08 us`、routed experts
  `50.56 us`、combine-add `4.32 us`、TP8 AR+等待 `41.92 us`。逐rank解释了AR值中
  包含arrival等待：DP0 shared约19--20us且routed约36--41us；DP1无shared但routed
  约41--51us；早到rank在AR内等待，AIter 8KiB TP8 AR本体仍约16us。
- 只在split DP0把shared expert放辅助stream与2-slot routed重叠，graph和France
  10/10均通过、hash不变，但热稳态约`74.86--75.19 tok/s`，相对串行
  `75.82--76.01`退化约1.1%；同卡CU/带宽竞争推迟collective arrival，完整撤回。
- 尝试在direct MoE下省略Top-K后的`topk_ids.fill_(-1)`时，France首轮立即错误为
  `[1,1,1,1,1,16,...]`。说明sentinel不仅限制direct kernel slots，还参与上游
  runner/dispatcher语义；该mask不能由`slot_begin/end`替代，改动已恢复且错误性能
  数据作废。

### M>2路径边界审计与Top-K producer sentinel反例（2026-08-26）

- 仓库不存在统一的“M>2不支持”。三个容易混淆的边界含义不同：gfx90a native
  grouped Top-K/router当前严格是`M=1,N=256`单CTA专核；已有`M=1..4`能力的是它前面的
  BF16 router projection wave64 GEMV。通用Triton router本身支持任意M，因此M>1只会
  失去native Top-K专核，并非correctness失败。
- Inkling gate的`M<=2`是CUDA资源分桶：权重切片驻留VGPR；`M=3/4`仍走专核但改为
  shared-memory staging，之后才回常规GEMM。这个按M改变staging/warp分工的思路可供
  gfx90a wave64 router复用，但CUDA实现不能直接移植。
- Inkling TP8 all-reduce的`M<=2`则是协议安全边界：v4省略exit barrier，依赖A/B
  双缓冲、reuse-distance=2及每次forward偶数次AR。它又依赖NVLS/multimem，既不能
  随意放宽到M>2，也不能直接移植ROCm。当前AIter gfx90a AR没有M<=2限制，小消息瓶颈
  是固定barrier成本。
- DSV4 direct FP4 MoE同样没有M>2禁令：M=1在kernel内量化并由shared memory复用；
  M>1先做外置per-token group-32 INT8 quant，避免每个expert CTA重复量化。大M
  grouped/MFMA路径主要受动态Mori token count与expert mask selector限制。后续可为
  native grouped router实现`grid.x=M`、每block一个token，并按M=1/2/4分别ABBA；它
  主要改善BS2/4，不会改善当前每个DP副本本地M=1的单请求路径。
- 尝试把split-MoE的`-1` sentinel直接写入通用Triton Top-K producer以消除独立fill。
  第一版同时跳过hash-router层fill，法国oracle首请求即EOS；修正为前三个HashTopK层
  保留fill后，首个请求正确，但随后graph replay持续生成错误序列。说明独立fill还在
  捕获图的复用buffer生命周期中提供稳定覆盖，不能只按producer类型局部消除。实验
  完整撤回，不采纳任何性能数据。

### split direct-FP4单行wave checkpoint（2026-08-26）

- 短上下文C4 indexer虽跳过logits，仍调用通用AOT Top-K的`seq_len<=topk` naive分支。
  实现过无LDS、256-thread的select-all HIP JIT专核；B=1/2/4、长度
  0/1/127/128/511/512以及raw/physical indices均逐元素一致。但graph内完整节点为
  现有`zeros+AOT`约`6.348 us`、新`empty+JIT`约`6.920 us`，退化约9%，完整撤回。
- direct FP4 gate/down此前固定每wave累加2个输出rows。真实TP8 split的四-slot shape
  中，`rows=1` gate约`29.33 -> 24.99 us`、down约`15.15 -> 14.93 us`；不同rows、
  blocks和waves的eager/graph输出均bitwise exact。rows4/8受VGPR压力明显退化，
  wave16的1024-thread block也退化，保持8 waves。
- 八卡graph ABBA：B1 rows1热态`78.20--78.38 tok/s`，A rows2约
  `76.19--76.35`，B2 rows1约`77.65--77.73`；B2相对A中位约+1.85%。法国oracle
  B1/B2合计20/20精确，所有256-token hash均保持`14593da264d38f29`。因此仅在
  TP8/DP2 split fast path且本地M=1时默认rows1；其它direct/prefill shape保持rows2。
  最终不显式覆盖环境变量的独立服务再次法国10/10、六轮hash一致，去首轮后稳定为
  `77.74--77.77 tok/s`。
- rows1的isolated gate在2/4 slots分别偏好128/156 blocks，但端到端约
  `77.47--77.56 tok/s`，低于继续使用208 blocks的B2，故不按owner缩grid。
- 尝试每次32-bit加载8个FP4 nibble、配对两次`sdot4`以替代两个16-bit load。
  micro在rows1 gate快约1--3%，且全部bitwise exact；完整服务却仅
  `76.88--76.98 tok/s`，低于rows1/load4，load8实现和开关已撤回。

### 双TP4的32并发SBO+multistream checkpoint（2026-08-26）

- 8 GCD继续拆成两个独立`TP4/EP1/no-A2A`副本，每副本16请求、16K token pool、
  graph tiers`1/2/4/8/16/20/24/32`。严格native AR、256-token、统一barrier的
  32总并发基线A1/A2热态trimmed分别为`753.58/751.38 tok/s`。
- 同时开启`--enable-single-batch-overlap`和`SGLANG_ROCM_USE_MULTI_STREAM=1`，
  让shared expert在ROCm辅助stream上与routed expert重叠。B1/B2热态trimmed分别
  为`818.41/810.17 tok/s`，配对均值相对A提升约8.2%。France fixed IDs四个服务
  合计40/40逐token精确；正式32并发请求均输出256 token并`finish=length`。
- 新增显式`SGLANG_DSV4_GFX90A_MULTI_REQUEST_THROUGHPUT_PROFILE=1`，只为多请求
  profile同时选择SBO和ROCm multistream，不改变默认BS1延迟配置。仅设置该profile
  的最终独立服务确认进程环境与server args均命中，France 10/10，32并发三轮热态
  `814.93--815.08 tok/s`。
- admission对照`CHUNKED_PREFILL_SIZE=4096`为`744.16 tok/s`，低于2048；更大首批
  prefill没有改善group wall。direct-FP4 grouped assignments从8改4仅约
  `758.17 tok/s`且出现一次慢态，改2退化到`725.49`；non-grouped direct仅约
  `528.17`，AIter/CKTile direct-off约`542.35 tok/s`。这些分支均不采纳，group8
  direct专核仍为M16默认。

### M16 dual-stream trace与grouped-FP4 persistent grid（2026-08-26）

- 原realtime MLP marker只接在`forward_normal()`；SBO+ROCm multistream的graph实际
  选择`forward_normal_dual_stream()`，因此旧16--24槽位只保留capture时间戳。将同一
  默认关闭marker接到dual-stream主分支，并用25/26记录alt shared分支后，M16 layer20
  France 5/5精确且取得有效replay分解。
- M16典型每层：router约`24--27 us`、Top-K约`12 us`、routed FP4约
  `230--365 us`、shared expert约`160--205 us`、join约`4 us`、combine/add约
  `4 us`、TP4 AR约`31--43 us`。shared已完全隐藏在更慢的routed分支后，routed FP4
  是当前最大单项；整层随路由分布约`0.8--0.93 ms`。
- 真实TP4/EP1 M16、83个unique experts的graph micro中，group8 gate/down从208 blocks
  的`335.13/211.42 us`降到624 blocks的`308.55/175.19 us`，全部逐元素一致。继续扫到
  gate1040/down1248仅为`300.73/167.36 us`，端到端收益已饱和且hash轨迹抖动更明显，
  因此选择624。
- 双TP4、SBO+multistream、32总并发ABBA：208-block A3热态约`812.32 tok/s`；624-block
  B1/B2 trimmed约`843.37/842.48 tok/s`，相对A约+3.7%，相对最初无SBO的`~752.5`
  累计约+12%。两轮624服务France合计20/20逐token精确，所有正式请求均256 token。
  B2有一次首个prefill仅`7.64 tok/s`并使整轮降到243，日志显示为新的JIT慢态；下一轮
  自动恢复843，故不计稳态但必须继续保留多轮trim。
- throughput profile现在将小M grouped gate/down默认grid设为624；普通延迟profile仍
  保持208，且两个grid均可分别用环境变量覆盖。

### 单模型TP8的1M-token多请求profile（2026-08-26）

- 双TP4副本继续作为短请求最大吞吐特例保留，但它会复制两份完整模型：每GCD仍持有
  约45.34GB权重，每副本16K pool后约余11.4GB。面向大并发与长上下文的正式拓扑改为
  单实例`TP8/EP1/no-A2A`，不再用模型复制掩盖KV容量。
- TP8每GCD权重约`26.80 GB`。`mem_fraction_static=0.96`、
  `MAX_TOTAL_TOKENS=1048576`、SWA ratio0.65下实际成功分配完整
  `1,048,576-token` pool；BS`1/2/4/8/16/20/24/32` graph总计约0.46GB，capture后
  仍余约`15.9 GB/GCD`。DSV4 allocator报告理论容量约1,155,840 full tokens，因此
  当前1M上限不是OOM边缘值。
- 基础TP8配置在全部8个graph tier执行France固定input IDs，合计`107/107`请求均与
  9-token expected IDs逐token精确且每tier输出唯一。BS32、256-token native AR冷态
  前两轮为`229.12/350.72 tok/s`，JIT完成后四轮稳定为
  `713.61/713.96/713.70/713.88 tok/s`；正式比较必须剔除首次shape编译。
- 仅开启`--enable-single-batch-overlap`与`SGLANG_ROCM_USE_MULTI_STREAM=1`，保持
  group8/208 blocks及同一1M pool，France在BS1/16/32合计`49/49`逐token精确。
  六轮为`779.15/782.90/766.96/768.47/768.60/768.55 tok/s`，trim约
  `771.19 tok/s`，相对TP8热基线约`+7.7%`。新增
  `SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=1`固化单模型TP8容量/吞吐配置；它与
  `SGLANG_DSV4_GFX90A_MULTI_REQUEST_THROUGHPUT_PROFILE=1`的双TP4特例互不替代。
- AIter BF16 tuned-config与M16 hipBLASLt projection在双TP4端到端仅约`+0.8%`，其中
  N1536 helper叠加收益约`0.1%`；复杂度不值，已从工作树撤掉，不混入TP8 checkpoint。

### TP8 BS32热点分解与负结果（2026-08-26）

- TP8/EP1、SBO+multistream、M32 realtime marker在layer20给出约`969 us/layer`：
  MHC pre约`54 us`，attention prepare约`242 us`，attention core约`55 us`，
  inverse-RoPE与`wo_a/wo_b`约`108 us`，attention后的MHC过渡约`49 us`，完整MoE
  约`458 us`。MoE内部router约`25 us`、Top-K约`12 us`、routed FP4约
  `346--376 us`、shared expert约`140--205 us`且被routed分支完全隐藏、TP8
  all-reduce约`33--41 us`。因此下一阶段第一瓶颈是routed FP4，attention总段与之
  同量级，collective不是第一优先级。
- M32真实routed shape中，将grouped assignments从8改4、gate/down每wave rows从2改4、
  blocks从208改832，standalone完整sort+gate+quant+down由`657.90 -> 517.09 us`
  （约`+27.2%`），输出逐元素bitwise exact；完整服务从约`768.6`升到约
  `786.5 tok/s`，仅`+2.3%`，尚不足以单独形成5% performance checkpoint。
- AIter custom all-reduce的8-rank graph micro：M16/128KiB约`22.6 us`，
  M32/256KiB约`32.3 us`。扫描one-stage/two-stage与8--64 blocks均无改善；源码实验
  已撤回。结合trace中AR仅约35us，继续调通信协议的收益上限太低。
- 尝试将C4 core-compressor join从indexer query前延后到query后。只捕获BS1/32的
  A2热态约`789.9--791.8 tok/s`；B1一度约`821.4`，但独立B2仅
  `786.0--791.3`，与A重叠，判定B1为服务状态高点而非可复现收益。France在A2/B1/B2
  合计`30/30`精确；依赖改动和开关已撤回。
- M32三个BF16 attention-prep投影的单GCD micro：独立顺序GEMM约`142.6 us`，三流
  并发反而约`210.6 us`，拼成一次N=4096 GEMM约`89.9 us`。融合结果对N512分支
  bitwise exact，N1536/N2048最大BF16差异0.5。完整服务France`10/10`正确，但六轮
  稳态除一次`822.6`高点外为`783--793 tok/s`，与未融合A/B重叠；额外约0.65GB/GCD
  的融合权重缓存不值，代码与开关已撤回。说明standalone GEMM节省被现有compressor
  后处理/多流尾部抵消，后续必须以graph replay分段而非裸GEMM决定是否接入。
- 为加快BS32专项A/B，graph tiers可临时只捕获`1,32`，capture约`6.3 s`、graph约
  `0.47 GB/GCD`；但多轮会偶发落入未捕获/新shape慢态（约238--364 tok/s），生产
  与最终验收仍需恢复`1/2/4/8/16/20/24/32`以覆盖admission和batch下降。
- 补齐此前缺失的TP8/EP2大并发对照：启用partial-EP、Mori world-size2、16-block、
  direct FP4并捕获BS1/32。France固定IDs`10/10`精确，但32并发三轮仅
  `394.36/398.90/399.12 tok/s`，约为EP1的一半；graph内存也由EP1约
  `0.47 GB/GCD`增至约`6.8 GB/GCD`（capture后仅余约9.5GB）。因此EP2不仅在
  BS1/2/4退化，在192 routed assignments的BS32仍不能摊平每层Mori固定成本，正式
  证伪为当前TP8吞吐方向。多并发若再引入EP，必须先改变dispatch/combine协议或采用
  更大的请求批次，而不是复用现有逐层Mori路径。
- 小M direct-FP4的两次group32 INT8 activation quant曾用已有HIP wave64专核替代
  generic Triton，并扫描每block `2/4/8/16` waves。两处真实M32 shape的量化结果均与
  Triton逐元素bitwise exact；单次quant由约`61--62 us`降到约`38--39 us`，完整合成
  routed stage由`629.76 -> 598.88 us`（约`+4.9%`）。但TP8服务级France固定IDs
  `10/10`精确后，32并发两组各6轮的主要稳态仅约`788--796 tok/s`（各有一次
  `824 tok/s`高点），对同tier Triton A约`790 tok/s`的trimmed收益只有约`0.5%`。
  量化已被graph并行/尾部隐藏，故撤回decode selector和多wave模板，不作为正式优化。
- M32 grouped direct-FP4增加独立micro harness并扫描row/wave/grid：`rows=4,waves=8`
  明确优于rows 1/2/3/5/6/8；gate/down同为832 blocks约`502.9 us`，down改1664约
  `496.0 us`，但完整TP8热态仍约`790--795 tok/s`，没有可复现端到端收益。尝试把
  gate/up的BF16 SwiGLU落地与后续group32 INT8 quant融合进同一8-wave CTA，INT8
  bitwise exact、scale最大差`7.9e-13`、最终BF16 bitwise exact，但LDS与两次barrier
  使完整stage由约`501 -> 620 us`，退化约24%，实现已撤回。
- 1M-token Unified Radix Cache暴露出独立的生产尾延迟bug：scheduler在每次fully-idle
  循环无条件执行全树`sanity_check()`；radix条目积累后，32并发轮会周期性由约
  `790 tok/s`跌到`361--363 tok/s`，退出栈显示八个scheduler都在tree walk。现将该
  debug walk只保留在`SGLANG_CHECK_KV_PAGE_INVARIANTS`或
  `SGLANG_INVARIANT_CHECK>0`下；不能复用strict-idle mem check，因为该分支后者默认
  为True。最终guard下TP8 BS1/32 graph的France固定IDs`10/10`精确；8轮32并发为
  `775.7/822.2/819.9/822.3/787.1/787.9/786.2/790.0 tok/s`，没有再出现约363的
  周期性腰斩。退出时scheduler栈也从radix `sanity_check()`变为正常的request broadcast
  wait，确认默认全树walk已被移出生产idle路径。这是尾延迟/可用吞吐修复；它不改变
  约`790 tok/s`的kernel热态中心。

### TP8数据缺口与FP4 unpack上界（2026-08-26）

- 对当前单实例TP8/EP1/no-A2A、1M-token、SBO+multistream profile做了口径审计：
  只有BS32具备当前配置的完整吞吐数据，稳态中心约`790 tok/s`。BS1/2/4/8/16现有
  `65.52/86.84/153.52/258.09/478.80 tok/s`来自旧8K pool、SBO-off服务，只能作为
  历史参考，不能填入当前矩阵。全部graph tiers `1/2/4/8/16/20/24/32`已确认capture，
  但仍需补逐tier replay命中证据与当前配置吞吐。
- France fixed-input oracle已覆盖基础全tier `107/107`，SBO覆盖BS1/16/32共`49/49`，
  最新radix guard覆盖BS1/32共`10/10`。不过当前1M/SBO checkpoint尚未记录相同
  128/256-token continuation在所有batch slot的逐token一致性，因此不能把短France
  通过扩大为长decode bitwise parity。下一次单服务补测应按
  `32 -> 1 -> 16 -> 2 -> 8 -> 4 -> 32`采吞吐，同时保存first-divergence与graph
  replay counter。
- rocprof的M32 grouped kernel显示gate/down分别约`253.2/252.8 us`，两次Triton量化
  各约`4.48 us`、final reduce约`5.16 us`。静态HSACO中gate编译体含约
  `192 v_perm / 432 v_and / 212 shift / 256 v_dot4`，说明packed-FP4在线解码具有
  显著VALU指令压力。
- 实验性地把E2M1权重离线展开成signed INT8 codebook，保持scale与sdot4数学不变。
  M32、A4/R4/832的严格ABBA为packed `504.40/504.36 us`，prepacked
  `443.06/444.36 us`，最终BF16逐元素bitwise exact；即完整routed micro上界约
  `12.0%`。但TP8每层全专家额外`768 MiB`，43层为`32.25 GiB/GCD`，会破坏1M KV
  容量，不能全量接入。
- 512-byte `__constant__` byte-pair LUT保持packed FP4且输出bitwise exact，但M32
  退化到`835.73 us`（相对现路径约`+66%`延迟），与CDNA2对lane-divergent constant
  lookup不友好一致，已撤回且不接服务。下一候选是每CTA 1KiB LDS LUT，必须同时测
  LDS bank conflicts；若仍不赢则转向共享SWAR selector，不能假定查表一定优于
  `v_perm`。
- 有限热专家cache每expert-layer额外约`3 MiB`；跳过近似均匀的前三个hash-router层，
  固定N=8/16/32/64约占`0.94/1.88/3.75/7.50 GiB/GCD`。是否值得必须先按实际
  `sorted_expert_ids`统计grouped weight-scan block命中率；双段N16若held-out命中率
  低于约46%，预计端到端连2%都达不到。若分布足够偏斜，优先评估w2-only cache，
  其单位显存收益约为w13的两倍。

### packed-FP4 CTA-local LDS unpack checkpoint（2026-08-26）

- constant-memory byte LUT因逐lane发散读取退化后，改为每个CTA用256线程初始化
  `uint32 pair_lut[256]`（1KiB LDS），每个packed `uint16`用两次LDS byte-pair读取
  产生四个signed-INT8 E2M1 code，再复用现有`s_dot4`累加。权重始终保持4-bit packed，
  不增加模型常驻显存；原三次`v_perm` decoder保留为默认/reference模板实例。
- M32、A4/R4/832严格ABBA为原路径`504.13 us`、LDS `351.25/351.58 us`、原路径
  `504.82 us`，完整gate+quant+down提升约30.3%，最终BF16逐元素bitwise exact。
  形状扩展也通过同一reference检查：M8 `175.25 -> 115.46 us`（约34.1%），M16
  `324.29 -> 251.34 us`（约22.5%），M32独立并行复测`500.74 -> 347.81 us`
  （约30.5%）。
- 通过默认关闭的`SGLANG_DSV4_GFX90A_FP4_LDS_UNPACK`接入non-MFMA grouped路径，
  且仅对`M<=64`生效；TP8 multi-request profile现默认开启，其他拓扑/普通profile
  保持关闭。graph capture `1/2/4/8/16/20/24/32`成功，graph约0.54GB/GCD，1M-token
  pool不变。
- 服务级correctness：候选在BS1/2/4/8/16/32的France fixed-input合计`63/63`
  逐tokenexact；独立完整tier服务的BS32再通过`32/32`。128/256-token并发hash仍
  存在基线已有的slot漂移，候选与LDS-off的主要hash集合高度重叠，不能宣称长生成
  bitwise parity已修复；这项基础correctness债务继续单独跟踪。
- 同机LDS-off A端热态主要为`788--792 tok/s`，另有`821 tok/s`高态；LDS-on B端
  首轮热态约`869.8--872.3 tok/s`。完整tier并耗尽动态BF16 M-shape JIT后，连续12轮
  为`876.65/874.86/876.30/875.48/878.02/873.42/874.20/874.64/876.16/876.12/878.03/875.37`
  tok/s，median `875.80`、trimmed mean `875.78`、零慢轮。相对约790中心保守提升
  约10.7%，是可提交的性能checkpoint。
- 早期候选与基线都出现约`243--381 tok/s`慢轮；退出日志显示AIter首次见到如M=11
  的shared-expert BF16 GEMM时即时选解/JIT。随着1--32实际active-batch shapes被缓存，
  候选最后12轮不再复现，因此该慢态不是LDS barrier、graph deadlock或radix walk。
- 在同一已热透的1M-token服务按`32 -> 1 -> 16 -> 2 -> 8 -> 4 -> 32`补齐当前配置
  吞吐矩阵（每tier两轮、256 token）：BS1 `73.81/74.45`，BS2 `99.85/99.69`，
  BS4 `176.79/176.91`，BS8 `313.58/314.04`，BS16 `565.57/581.14`，BS32端点
  `870.86/875.14/903.78/875.34 tok/s`。BS32的`903.78`视为高态，稳态中心仍约
  `875--876`；tier下降后回到BS32没有性能塌陷。

### TP8吞吐预算、CK对照与decode-TBO证伪（2026-08-26）

- 当前1M-token、SBO+multistream、LDS-unpack服务的统一矩阵如下。BS1--16取同一热服务
  两轮均值，BS32取12轮median：

  | tier | aggregate tok/s | amortized step ms | per-request tok/s | BS1 linear efficiency |
  |---:|---:|---:|---:|---:|
  | 1 | 74.13 | 13.490 | 74.13 | 100.0% |
  | 2 | 99.77 | 20.046 | 49.89 | 67.3% |
  | 4 | 176.85 | 22.618 | 44.21 | 59.6% |
  | 8 | 313.81 | 25.493 | 39.23 | 52.9% |
  | 16 | 573.36 | 27.906 | 35.83 | 48.3% |
  | 32 | 875.80 | 36.538 | 27.37 | 36.9% |

  BS32达到`1500 tok/s`需要step从`36.538 -> 21.333 ms`，即净减`15.205 ms`
  （41.6%）；按43层折算由约`849.7 -> 496.1 us/layer`，需净减约
  `353.6 us/layer`。
- LDS后的layer-20/M32 marker中位数为MHC-pre约`51.8 us`、attention prepare约
  `239.6 us`、attention core约`42.9 us`、attention output约`104.5 us`、MHC transition
  约`50.6 us`、MoE约`358.5 us`。两次TP8 collective合计通常约`65--73 us/layer`，
  不是第一瓶颈。目标层预算几乎等于当前完整attention侧，因此必须隐藏大部分MoE，
  同时继续降低M16/M32的权重扫描成本。
- 用本地Composable Kernel `ckProfiler`扫描真实M32 BF16投影shape，CK最佳分别为
  `(N,K)=(1536,4096) 74.97 us`、`(2048,4096) 81.12 us`、`(512,4096) 62.12 us`、
  `(64,4096) 59.14 us`、`(8192,1024) 34.51 us`、`(4096,2048) 48.13 us`；同机
  `torch.nn.functional.linear`分别约`35.66/36.91/28.34/35.47/28.62/30.03 us`。
  CK在全部目标shape上均更慢，因此当前不接入regular BF16 CK路径。
- 实现过一个仅由`SGLANG_DSV4_GFX90A_DECODE_TBO=1`开放的eager诊断原型：把BS32拆成
  两个M16 child，在attention stream与MoE stream间用逐层event形成流水。France
  `32/32`逐token exact，说明基础依赖没有立即死锁；但它关闭decode graph且使用
  非融合MHC。严格matched、完整256 token、`ignore_eos=true`的普通eager基线为
  `191.22 tok/s`，TBO原型仅`104.15 tok/s`，至少回退45.5%。原型还缺少跨stream
  temporary的`record_stream`/keepalive，存在allocator复用风险，不能作为可保留路径。
- 该原型退化有结构原因：两次M16会重复读取投影与expert权重；LDS FP4 micro中
  `2 x M16`约`502 us`，而一次M32约`351 us`。由当前BS16约`581 tok/s`推导，即使
  两阶段attention/MoE完美重叠，简单2xM16流水的绝对上界也仅约`1.15--1.16k tok/s`，
  现实约`1.0--1.1k`，不足以单独达到1500。只有真实stage micro的两phase时间降到
  `<=575 us/layer`（约1294 tok/s）才值得重新接模型/graph；原型代码因此撤回，只保留
  数据和设计结论。
- 当前correctness证据边界保持不变：LDS微核M8/M16/M32最终BF16 bitwise exact，France
  覆盖当前tier，但128/256 token仍有基线既存的跨slot greedy漂移。后续性能改动至少要
  做France全tier，并比较长序列first-divergence；最终应增加固定continuation的
  teacher-forced top-20 logits oracle，而不是只用最终生成hash二分。

### LDS后的grouped geometry复扫（2026-08-26）

- 发现TP8 profile此前只默认开启LDS unpack，并没有进入四卡multi-request分支的
  `624`-block默认值；实际仍是代码默认`assignments=8, rows=2, blocks=208`。LDS改变了
  kernel的寄存器/LDS占用后，旧几何不是最佳点，因此重新扫描M32的
  `assignments=4/8`、`rows=2/4`、`blocks=416/624/832`。
- 全部LDS候选相对同几何packed reference最终BF16逐元素exact。关键micro结果：
  production-near `A8/R2/B624=411.24 us`，`A8/R2/B832=423.32 us`；最佳为
  `A4/R2/B832=329.79 us`，相对A8/R2/B624节省`81.45 us`（19.8%）。A4/R2/B416
  也为`330.74 us`，而R4各点约`339.92--347.81 us`，因此选择A4/R2/B832以保留
  较大的跨CU grid。
- 服务B1显式设置A4/R2/B832，完整graph tiers capture成功，France BS1/2/4/8/16/32
  合计`63/63` exact。BS32、32请求各256 token、`ignore_eos=true`连续12轮为
  `913.86/914.75/911.82/911.86/910.06/911.84/914.45/910.57/912.50/913.47/910.79/911.14`
  tok/s，median `911.85`、trimmed `912.23`。
- A2回到TP8 profile原默认A8/R2/B208，同样先做全tier France `63/63`，八轮BS32为
  `843.79/844.48/847.98/844.99/846.11/847.27/847.24/848.29 tok/s`，median
  `846.68`、trimmed `846.34`。B2再次使用A4/R2/B832，全tier France仍`63/63`，八轮
  为`916.28/916.46/916.39/915.13/914.60/916.98/910.74/916.06 tok/s`，median
  `916.17`、trimmed `915.82`。配对B2/A2提升约8.2%；相对此前更高的完整tier
  LDS checkpoint `875.78`仍提升约4.6%。ABBA方向一致，说明收益来自几何而非单次
  服务高态。
- B1的BS32/256-token完成hash有7种（主两种各12/11 slot），全部长度256；这与当前
  TP8基线已有的跨slot greedy漂移同类，不能据此宣称长bitwise parity。短France和
  micro exact均通过，但固定continuation logits oracle仍是独立correctness债务。
- TP8 profile现默认导出`SGLANG_DSV4_GFX90A_FP4_GROUPED_DECODE_ASSIGNMENTS=4`及
  gate/down blocks `832`；其他TP4/普通profile继续保持原A8及208/624默认值，显式环境
  覆盖仍优先。

### 真实M32 stage overlap probe（2026-08-26）

- 为排除旧eager TBO被non-fused MHC混淆，曾加入默认不可达的一次性诊断probe，在
  layer 20的真实`mark(1)`/`mark(6)`边界捕获输入，直接重复生产attention和MoE闭包；
  使用两条stream、显式event、`record_stream`与平衡ABBA，先以RCCL关闭custom AR和
  decode graph测试M32。probe不包含约102us的MHC固定段。
- France BS32在probe过程中仍为`32/32` exact；8个rank的attention/MoE/serial/overlap
  输出全部逐元素exact且finite。rank-max median为attention `1.915 ms`、MoE
  `1.541 ms`、serial `3.253 ms`、overlap `3.396 ms`；空fork/join约`0.051 ms`。
  各rank overlap相对serial的saved time均为负，范围`-0.155..-0.123 ms`，MoE并发
  slowdown约`1.008--1.060x`。
- 因此当前真实M32 attention与MoE在双流/RCCL下没有形成GPU计算重叠，collective和CU
  资源使两段近似串行，且额外event/队列带来退化。结果远低于继续门槛（至少20% saved、
  总层预算<=575us）；不再投入decode-TBO graph化，也无需冒险测试custom AR的并发
  scratch/reentrancy。诊断helper与forward hook已撤回，只保留数据结论。

### LDS几何尾扫与PP2/TP4 oracle（2026-08-26）

- 在A4/R2基础上分别固定down或gate扫描`208..2080` blocks及4/8 waves。最佳micro为
  `waves=4, gate_blocks=1664, down_blocks=1664`约`322.03 us`，相对已交付的
  A4/R2/waves8/B832约`329.8--330.5 us`只再省约2.4%；不足端到端5%门槛，且runner
  目前不暴露waves变量，因此不启动服务、不增加新开关。
- 尝试单模型`PP2 x TP4`，而非两个重复TP4副本：每个GCD只加载半数层的TP4 shard，
  `pp_max_micro_batch_size=16`、global BS32、PP async depth 0。模型/graph正常启动，
  1M-token pool保持不变；每GCD模型约`22.6--23.2 GB`，capture后仍余`28--30 GB`。
- France BS1/2/4/8/16/32合计`63/63` exact，但BS32/256-token四轮为
  `749.24/747.02/358.58/748.23 tok/s`，正常中心约`748`且有一次慢态，明显低于当前
  TP8约`912--916`及预设`830`停止门槛。现有PP microbatch调度与TP4更宽权重扫描没有
  形成收益；不继续扫async depth。临时PP harness接线已撤回，仅保留实验记录。

### A4几何后的trace与attention-concat复测（2026-08-26）

- 在新默认A4/R2/B832、M32、layer20重新抓取520个rank-sample，去掉每rank前两次后
  marker median：coarse=`48.32/1.44/196.48/32.80/106.80/49.44/316.24 us`，
  prepare=`1.44/1.44/45.60/16.32/24.64/9.76/48.96/46.88/1.28 us`，MoE=
  `1.44/23.84/12.16/228.48/4.00/4.16/1.44/1.44/37.92/1.44 us`。marker总计约
  `751.5 us/layer`；目标1500 tok/s约需`496 us/layer`，仍需约255us结构性降幅。
- 恢复此前M32 C4 attention-prep concat：将BF16的N1536/N2048/N512三个投影拼成
  N4096，20个C4层额外约625MiB/GCD。当前graph micro独立投影`103.90 us`，concat
  `51.98--52.70 us`；N512结果exact，N1536/N2048因GEMM归约顺序最大BF16差约
  `1.0/0.125`。
- 新A4服务全tier France仍`63/63` exact，但BS32八轮仅
  `913.29/916.01/907.83/911.00/915.18/912.58/909.47/913.52 tok/s`，median
  `912.94`、trimmed `912.51`，相对当前默认B2 trimmed `915.82`无收益。concat推迟了
  原本并行的两条compressor支路，裸GEMM节省没有缩短graph critical path；实现和开关
  再次撤回。

### TP8 K256 down的8-lane subgroup checkpoint（2026-08-27）

- grouped down原来固定使用16-lane subgroup，但TP8的down shard只有K=256，即8个
  group-32 dot；lane 8--15不进入group loop，且旧offset-8 reduction只加入零。现按
  `min(K/32,16)`选择subgroup width：K256用8 lanes、每wave 8个subgroup，K512及以上
  保持16 lanes；增加K整除、power-of-two和wave64整除静态断言。非零项加法树仍为
  offset `4/2/1`，数学顺序不变。
- 生产A4/R2/W8/B832/LDS micro全部相对原LDS reference最终BF16 exact：M8
  `100.82 us`、M16 `164.68 us`、M32 `292.49 us`。M32相对同几何旧约
  `329.8--330.5 us`节省约38us（11.5%）。K512兼容实例也通过HIP syntax compile。
- 完整graph tiers capture成功，France BS1/2/4/8/16/32合计`63/63` exact。首次服务
  前两轮和中途一轮因新动态M实例JIT为`398.6/395.1/252.0 tok/s`；热透后连续12轮为
  `963.25/970.10/969.86/969.04/969.77/969.86/969.36/970.25/970.39/966.88/970.18/966.61`
  tok/s，median `969.82`、trimmed `969.19`，零慢轮。相对上一A4 checkpoint B2
  trimmed `915.82`提升约5.83%。
- BS32/256-token完成hash仍为改动前完全相同的7种及相同slot计数：主两种12/11，
  其余4/2/1/1/1，全部长度256。结合micro exact和France全tier，说明此改动没有引入
  新分叉；原有TP8跨slot greedy漂移仍作为独立correctness债务。
- 后续尝试将LDS path从“先解包8组权重、再逐assignment dot”改为逐j
  `decode -> sdot4`软件流水。M32输出仍exact，但完整micro由`292.49`退化到
  `1313.46 us`。汇编虽使gate VGPR从94降至68、最长LDS read burst从24降至4，却把
  vector global load拆成大量scalar load，并使`s_waitcnt`约122增至233、
  `s_and_saveexec`约24增至136；控制流/等待成本压倒寄存器收益。P1已完整撤回。
- 另外三类后续候选均未进入正式配置：删除LDS lookup后的16-bit mask为`297.66 us`
  （退化约1.8%）；4份、257-stride skew LUT为`306.25 us`（退化约4.7%）；现有
  MFMA32 gate/down直接用于M32约`1.26--1.44 ms`且相对sdot reference最大BF16差达
  4--32，均立即停止并撤回。
- 稀疏几何micro中A2/gate-rows2/down-rows1/B1664可到`268.92 us`，但完整服务France
  全tier虽`63/63` exact，稳态只有约`946.7--949.1 tok/s`，低于A4默认约969.8；
  sorter/padding与更大grid抵消裸kernel收益。保持A4、只设down-rows1的micro约
  `275.8--276.3 us`，服务稳态约`965.8--968.9 tok/s`，同样未超过默认。两项均不保留。

### TP8 BS32 expert occupancy与MFMA上界（2026-08-27）

- 对`M=32, topk=6, 256 experts`按每token内部不重复、expert近似均匀建模，单expert
  occupancy为`Binom(32,6/256)`，期望`k=0..6`的expert数约为
  `119.85/92.05/34.24/8.22/1.43/0.192/0.0208`，即约`136.15`个active experts。
  assignments中约47.94%属于singleton expert，48.51%属于occupancy 2--3，
  occupancy>=4仅3.55%。M16实测约83个unique experts，与均匀模型的80.84接近，
  因而该模型足以解释当前geometry结果。
- assignment block扫描数期望为：A1=`192`、A2=`146.23`、A4=`136.36`、A8及以上
  约`136.15`。当前A4距理论最少weight scan仅约0.16%；A2多约7.23%，A1多约40.8%。
  因此继续增大assignment不能减少有意义的权重扫描，只会扩大寄存器数组和padding。
- TP8每个expert每次gate/down合计约读取`3.1875 MiB`的FP4权重与scale，A4约为
  `455.8 MB/layer/rank`。当前M32 routed micro `292.49 us`对应约`1.56 TB/s`，已接近
  单GCD约1.6384TB/s的名义HBM带宽；除权重扫描外的理论余量约14us（约5%）。
- 现有MFMA32在M32约`1.26--1.44 ms`并非简单调参问题：每个active expert通常只有
  1--3行，却固定执行16/32-row tile，约94--97%为padding；split-K LDS partial、归约
  和双barrier又放大固定成本。其FP4 codebook/scale mapping可复用，但全量M32 MFMA
  路径不值得继续接服务。
- 唯一尚合理的occupancy-aware候选是singleton A1与multi A4分流，但它不减少权重
  流量并增加partition metadata和graph kernel launch；结合A2 micro更快、服务却仅
  `946.7--949.1 tok/s`的反例，预计端到端收益不足1--2%，不列为当前优先方向。

### TP8真实BS32路由热度与后续weight-reuse探针（2026-08-27）

- 使用SGLang `stat` expert-distribution recorder记录三轮32请求、每请求256-token的
  native AR，共得到768个完整BS32 decode passes。TP8 recorder的logical count在每个
  rank上已按world-size求和，分析时除以8；每pass/layer严格恢复为`32*topk6=192`
  assignments。此次harness的32个请求使用相同基准prompt和独立cache salt，因此该数据
  精确描述当前目标benchmark，但不能外推为任意自然流量。
- 与早先均匀路由估计不同，实际每pass/layer平均只有`39.07`个active experts；A4的
  weight blocks平均`61.46`，A8=`46.00`，A16=`40.67`，A32=`39.07`。这说明同批请求
  的路由高度相关，A4仍会对同一expert重复扫描约36%的权重。
- 用偶数decode pass选择每层热expert、奇数pass做held-out验证，Top-N覆盖的A4 block
  scan比例为：N8 `31.42%`、N16 `45.06%`、N24 `54.48%`、N32 `61.99%`、N48
  `72.91%`、N64 `80.56%`、N96 `90.43%`、N128 `95.47%`。assignment覆盖率N64为
  `82.91%`、N128为`96.06%`；前三个hash-router层明显更分散。
- 选择性INT8预展开cache虽有热度，但在当前LDS decoder后已无计算收益：同一M32/A4/
  R2/W8/B832 stage micro中，packed-LDS gate `155.44 us`、全prepacked gate
  `246.02 us`；packed-LDS down `135.68 us`、prepacked down `143.65 us`，输出均
  bitwise exact。完整prepacked stage `392.93 us`，而packed-LDS为`292.61 us`。
  因此不以额外4--8GiB/GCD换取prepacked hot cache。
- 使用一个真实occupancy profile构造相同expert count的synthetic micro：A4/R2/W8/
  B832为`192.31 us`；A8/R1/W8/B832为`207.07 us`，调到A8/R1/W4/B1664为
  `192.93 us`、B2080为`189.02 us`。A8虽减少约25%的global weight scans，却因8份
  accumulator的VGPR/控制成本只留下约1.7%收益。下一候选是每wave将一个row的packed
  gate/up一次解包到LDS，再按两个A4 chunk计算，以一次global scan同时避免acc8常驻；
  仅当gate micro相对A4至少提升15%才进入服务。
- 独立`V_MFMA_I32_4X4X4I8` A4 prototype完成了正确lane mapping，结果finite、相对
  LDS reference `max_abs=0.25`、relative-L2约`1.7e-5`；但gate耗时`4258.30 us`，
  对比A4/LDS `158.50 us`慢26.9倍。K4096下每wave需约2048条two-pass MFMA，该方向
  已完整撤回，未接selector。
- 为消除A8的8份accumulator常驻，依次验证了三种单次global-scan staging：每wave
  8KiB decoded gate/up LDS、每8-lane subgroup 256B down LDS，以及转置
  `[wave][side][j][group]`并只stage 124/128 groups的31KiB gate LDS。全部输出对A4
  BF16逐元素exact，但gate分别为`301.50/245.20 us`（A4约`103.4--103.6 us`），down
  `111.59 us`（A4 `83.69 us`）。后一版保持32KiB/CTA、避免bank conflict，并对66.4%
  second-chunk-empty blocks做wave-uniform skip，仍慢2.37倍；显式global->LDS写、二次
  LDS读与同步成本压倒VGPR收益，三项均撤回。
- 给服务补过临时decode-wave接线，测试原生A8/R1/W4/B2080（只捕获BS1/32）。France
  BS32为`32/32`逐tokenexact且输出唯一；动态shape热透后BS32/256-token两轮为
  `945.12/946.70 tok/s`，低于A4 checkpoint约`969.8 tok/s`。因此即使真实路由相关、
  A8 micro偶尔快约2--5%，sorter/grid与完整graph仍使端到端退化约2.4%；wave接线和
  A8配置均撤回，保持A4默认。

### TP8/PP2归因与C4 dual-compressor探针（2026-08-27）

- 当前最后一个通过完整correctness gate的BS32 checkpoint仍为TP8/EP1/no-A2A、
  A4/R2/B832 LDS unpack与TP8 K256 8-lane down：12轮中位`969.82 tok/s`、trimmed
  `969.19 tok/s`，France全tier `63/63`。本节后续结果均未替代该checkpoint。
- PP2×TP4约`748 tok/s`并非pipeline没有运行：global BS32对应`42.78 ms/step`，每级
  M16 service time约`21.39 ms`。PP0为21层且包含3个较便宜hash-router层，PP1为22层
  并带HC head/norm/LM head；TP4 shard又是TP8的两倍，且该profile关闭scheduler
  overlap。即使改为22/21反向切分，估计也只回收约4--5%，仍明显低于TP8。PP2的价值
  是容量：最小stage约可容纳`2.57M` tokens，约为当前TP8正式1M pool的2.5倍。
- C4 core/index compressor的原路径为两个stream、共4个postprocess launch；原型HIP
  dual kernel在M32把独立postprocess由约`132--143 us`降至`7.2--7.5 us`。使用刻意
  不同的core/index out-loc地址域后，core/index state、临时输出和两种cache仍全部
  byte-exact。乐观E2E上限约`969 -> 1055 tok/s`，但原两支多被main stream隐藏，真实
  收益可能远小于8.9%，必须测join exposed tail。
- 生产ABI审查确认core Unified-KV使用`unified.c4_out_loc`，index cache使用原始
  `c4_out_loc`；两者不可复用。state均为FP32，core cache为BF16/page1，index cache为
  FP8-preshuffle/page64；FP4 indexer与非Unified-KV必须拒绝。
- 为让dual与reference逐位一致，曾把公开index epilogue抽成共享device helper；该
  refactor即使dual开关关闭也会令首次真实请求退出，而standalone decode oracle无法
  暴露这一生产prefill问题。共享header已恢复到HEAD逐字节状态，dual生产接线和开关
  也已完整撤出默认代码，只保留未跟踪的独立原型/benchmark供后续重新设计。
- 最初用临时`urllib.request.urlopen()`检查服务时继承了代理环境，代理返回`502`，一度
  被误判成八rank设备fault。前台复查确认scheduler与GPU进程始终存活、`/health=200`；
  改用正式harness同样的`ProxyHandler({})`后，恢复原header/撤出dual接线的服务France
  BS1=`1/1`、BS32=`32/32`逐token exact。以后所有本机HTTP oracle必须显式禁用代理，
  不得把代理层502当作设备或kernel故障。
- 随后用正常`start`流程（显式`NO_PROXY`、graph tiers仅`1/32`）复测，日志确认八个
  scheduler与detokenizer均已freeze GC。France BS1/BS32合计`33/33`逐token exact；
  BS32/256-token八轮为`937.07/947.30/943.07/948.23/952.93/949.33/947.44/947.23`
  tok/s，中位约`947.37 tok/s`。该次低于历史完整tier checkpoint约2.3%，尚不足以
  宣称基线回退；长输出跨slot hash仍呈现既有多hash债务，但France没有新增漂移。
- 单独拆出的C4 core-only HIP后处理同样byte-exact，reference约`13.56--17.90 us`、
  fused约`7.87 us`，每个C4层仅节约约`6--10 us`，端到端乐观收益不足1%，因此未接
  生产服务并已删除原型。dual探针的主要micro时间显然来自index分支，但其原路径与
  core分支并行，仍需以join exposed tail而非两支耗时简单相加判断价值。

### TP8 BS32精确expert occupancy数据（2026-08-27）

- 对rank0 recorder中的768个完整BS32 passes重新统计，每pass/layer均严格恢复192个
  assignments。可复现分析脚本为`analyze_tp8_bs32_expert_occupancy.py`，逐层CSV为
  `tp8_bs32_expert_occupancy.csv`。
- 每pass/layer平均`39.065`个active experts。occupancy为1/2/3/4的active expert数
  分别为`7.097/5.498/5.005/9.722`；occupancy<=4占active experts约69.94%，但仅承载
  37.50%的assignments。5--8、9--16、17--32分别承载23.30%、20.10%、19.11%。
- 固定tile统计：A4平均61.463次weight scan、容量利用率78.10%；A8为46.001次、
  52.17%；A16为40.668次、29.51%；A32为39.065次、15.36%。A4到A8可减少25.15%
  scan，A8到A16仅再减11.59%，因此下一原型最多验证真实A8，不能直接扩大到M16/32。
- CDNA2只有I8 `16x16x16` MFMA而无`16x16x32`；真实A8 MFMA必须每K32发两条指令，
  并用多CTA/expert维持CU占用。独立micro stop-gate设为完整stage低于`248 us`（相对
  A4约292us至少15%）；未达标则不得接selector或服务。
- 独立A8 MFMA16 gate原型使用真实`39 active experts/192 assignments`、A4为61次scan、
  A8为46次scan。K32 I32 dot oracle逐元素exact；完整BF16输出finite，相对A4 reference
  `max_abs=0.0625`、mean abs约`2.94e-6`、relative-L2约`4.82e-6`。但A4 gate仅
  `103.232 us`，A8 MFMA16为`303.600 us`（慢2.94倍），远未达到`<=0.8x`门槛。
  因此没有实现down、没有接selector/graph/service，三个独立原型文件已删除。这个结果
  证实即使移除MFMA32的split-K/LDS partial税，16-row固定M tile在真实A8 occupancy下
  仍无法战胜当前wave64 LDS-sdot；下一步不再沿用固定16-row MFMA做BS32 routed MoE。

### TP8 BS32多样输入occupancy采集工具（2026-08-27）

- 旧的768-pass recorder来自32份相同prompt加独立cache salt，能精确描述固定France类
  benchmark，却不能代表32条自然请求。新增固定语料
  `.agents/memory/dsv4_tp8_diverse_32_input_ids.json`：32条中英文、数学、代码、系统与
  科学问题均保存为官方DSV4 chat格式的固定`input_ids`，同时保存tokenizer JSON SHA256，
  避免后续tokenizer或wrapper变化污染路由比较。
- 独立client `scripts/rocm/collect_dsv4_tp8_expert_occupancy.py`通过record endpoint启动
  `stat`记录，统一barrier发出32条请求，生成`32 warm + 128 window + 8 tail`个原生AR
  token，停止并dump recorder；它强制检查每请求长度，并用第0条请求的France首9 token
  oracle做correctness gate。服务需以`EXPERT_DISTRIBUTION_RECORDER_MODE=stat`和至少
  约170-pass buffer启动。
- `analyze_tp8_bs32_expert_occupancy.py`现按mtime选择最新dump，只保留checksum严格等于
  `BS32*topk6*43*TP8`的完整decode pass，丢弃前32个warm pass并分析后128个。CSV/JSON
  同时输出hash layers 0--2、learned layers 3--42、全层及可选逐层的occupancy histogram，
  A1/A2/A4/A8 scan、padding、capacity utilization，以及按当前H4096/I256/group32推导
  的逻辑weight bytes与bytes/useful-assignment。该bytes是kernel逻辑读取估计，不冒充
  hardware-counter测得的HBM traffic。
- 现有同prompt dump已通过CPU验证：128-pass窗口的每pass严格恢复`32*6*43`
  assignments，CSV/JSON可生成、脚本可编译。多样输入正式数据尚需下次TP8 recorder服务
  运行后产生，不能用旧dump替代。

### TP8 BS32 benchmark口径与HIP attention多流接线（2026-08-27）

- 历史`969.82 tok/s`与后来约`947 tok/s`并不是同一输入：scheduler日志的pending-token
  显示历史轮为每请求11-token France输入，后者脚本为16-token Explain-2+2输入；两者
  不仅prefill长度不同，后续greedy路由轨迹也不同。完整graph tiers、同一HEAD下，16-token
  输入八轮约`941.67--953.83 tok/s`；固定11个France `input_ids`六轮为
  `967.23/991.08/991.54/991.78/988.08/991.41`，median `991.25`、trimmed `990.53`。
  因此没有生产kernel回退，正式性能对比以后必须固定input IDs，不能混用这两组prompt。
- 只读审计发现`DeepseekV4Model`虽为HIP创建两条alt streams，`MQALayer.__init__`却只在
  CUDA/NPU条件下保存它们，导致现有`_forward_prepare_multi_stream_hip()`不可达。串行
  C4 prepare约`196.48 us/layer`，主要为wqkv_a `45.60`、qnorm `16.32`、wq_b
  `24.64`、store `9.76`、indexer `48.96`、core compressor `46.88 us`。
- 临时补上HIP alt-stream接线后，完整tiers graph稳定捕获，France全tiers仍`63/63`
  exact；固定11-ID France长生成六轮为`966.60/988.92/989.36/989.42/990.42/988.88`，
  trimmed `989.14 tok/s`。回退接线后的A2同样France `63/63`，六轮trimmed
  `990.45 tok/s`；A1/B/A2分别`990.53/989.14/990.45`，B稳定约慢0.13%。说明三条
  BF16 projection/compressor流竞争同一GCD的HBM/CU，理论重叠未缩短rank-max tail。
  接线已撤回且未提交生产代码；若再做此方向，必须先在独立graph micro把C4 prepare压到
  `<=156 us`才值得service，`<=118 us`才可能单项达到5%。

### gfx90a AIter BF16 M32定向调优（2026-08-27）

- TP8 graph capture日志显示BF16 `M32,N256,K4096`与`M32,N512,K4096`没有gfx90a
  tuned row，均回退AIter `torch solution:0`。前者主要为learned router；后者同时被
  C4 index compressor和TP8 shared-expert gate/up复用。AIter BF16 tuner在单GCD、
  `cu_num=104`上得到hipBLASLt N256 solution3992=`13.8415 us`、N512 solution5042=
  `15.4779 us`，tuner err_ratio均为0。
- 独立CUDA/HIP graph实测并未复制tuner的裸kernel数字：N256 torch/tuned为
  `23.20/24.64 us`，故立即排除；N512为`42.40/24.48 us`，约快42.3%。五组随机输入
  全部finite且所有元素满足`abs(error)<=0.05+0.05*abs(ref)`；首次replay之后连续12次
  graph replay bitwise稳定，最大BF16差`0.125`。
- 将N512 row作为全局AIter配置加载会同时改变compressor与shared expert。France全tiers
  `107/107` exact，但六轮固定11-ID France仅trimmed `980.08 tok/s`，相对A基线约
  `990.45`退化1.05%；更高CU占用破坏shared/routed overlap。
- 随后只在`linear_bf16_fp32`的compressor调用点定向solution5042，明确阻止shared
  expert命中。France仍`107/107` exact，但八轮为
  `958.53/984.37/985.86/983.52/986.05/982.74/983.77/984.49`，trimmed
  `984.13 tok/s`，仍退化约0.64%。两个生产候选均已撤回，未提交硬编码solution；
  结论是单kernel graph快不能代表C4 compressor完整链或层级critical path更快。

### C4 dual preshuffle与TP8后续结构边界（2026-08-27）

- C4 core/index dual HIP原型完全私有复制production epilogue，没有再改公开
  `fused_norm_rope_v2.cuh`。正式AIter index cache使用tile-16 preshuffle；最初普通布局
  oracle不足以证明长上下文正确，修正为与public `kPreshuffleSize=16`相同的地址公式后，
  连续四个独立进程的core/index tmp、两套state、BF16 core cache、FP8 index cache及
  scale均byte-exact。串行reference到dual的trimmed结果为`31.552->7.330`、
  `30.332->7.321`、`30.004->7.294`、`32.738->7.308 us`，每C4层省约23--25us。
- 默认关闭的临时生产接线仅命中HIP、strict decode、M32、C4、Unified-KV、非FP4-indexer、
  无CP路径；prefill和其他tiers保持原实现。修正preshuffle后完整graph捕获成功，France
  tiers 1/2/4/8/16/20/24/32合计`107/107` exact。4096-token混合input IDs跨过sparse
  indexer边界，dual与baseline的16个completion IDs完全相同，SHA均
  `a563b647156bc5fb`。
- 端到端固定11-ID France的B八轮trimmed=`987.37 tok/s`；回程A同机八轮trimmed=
  `987.11 tok/s`，仅+0.026%，属于噪声。micro节省在完整graph中被其他工作隐藏，生产
  接线已撤回；保留三个未跟踪standalone原型供后续trace研究，不作为性能成果。
- 短请求在`c4_seq_len<=index_topk`时已经跳过Q/weight/logits，但仍写index cache。
  依据请求最终horizon跳过cache只对禁用radix/session/HiCache复用的terminal graph安全；
  通用实现必须区分`dense_terminal/dense_growing/sparse`三variant或维护
  `index-valid-through`并在复用时reprefill。裸上限约1--3%，暂不为benchmark引入不通用
  的shortcut。
- 两次TP8 output all-reduce合计约65--73us/layer，全部删除的E2E上限也仅约
  `1085--1097 tok/s`。普通hidden不能永久RS，因为expert gate/up在SwiGLU前需要完整H。
  中期5--10%候选是仅让MHC residual按H分片：MoE边界`RS->sharded MHC->row-sharded
  wqkv_a->AR1536`；attention边界`RS->sharded MHC`后并行`AG4096`与row-sharded
  router `AR256`。必须先做layer20真实tensor oracle，并要求每边界slowest-rank至少省
  20us、1000次graph replay稳定，不能直接改模型。

### TP8基线复核、PP容量定位与sort8 hybrid反例（2026-08-27）

- 固定11-token France input IDs、BS32×256、full graph tiers、1M token pool的当前
  TP8/EP1/no-A2A热态可信中心为`987--991 tok/s`。当前服务八轮去掉首个冷态后
  trimmed=`987.11 tok/s`，近期另一独立服务trimmed=`990.53 tok/s`；约0.4%的
  跨服务波动要求后续只用同输入、同机ABBA判定。France全tier `107/107`逐tokenexact，
  4096-token mixed-ID oracle的completion SHA仍为
  `a563b647156bc5fb`。16-token Explain-2+2的`947--954 tok/s`不可与此口径混用。
- PP2×TP4功能与correctness均成立，正常BS32中心约`748 tok/s`，但比当前TP8低约
  24%；它的价值是把估计KV容量提高到约`2.57M tokens`，而不是当前1500 tok/s吞吐
  路线。按`987.11 tok/s`计算，step约`32.42 ms`，目标1500需`21.33 ms`，仍需净省
  `11.09 ms/step`（34.2%，约258us/layer）。配置审计未发现遗漏的已验证正收益开关。
- routed-FP4尝试过sort8后按第5项valid分流：occupancy 1--4由A4 light kernel处理，
  5--8由A8 heavy kernel处理，两核共享gate输出/down partial，down只做一次最终reduce。
  真实recorder pass42/layer20（active experts=40）中，生产A4/LDS B832完整stage七轮
  median=`194.773 us`；hybrid的heavy B208/416/832分别为
  `292.245/278.757/245.610 us`，最佳仍退化26.1%。所有最终BF16输出逐元素exact。
  双kernel重复遍历sort blocks并各自初始化/同步1KiB LDS LUT，开销超过减少25.15%
  weight scan的收益；未接生产selector，原型完整撤回。若重访weight reuse，必须在
  单kernel内动态复用且不让light block承担A8的VGPR常驻，不能再用双launch分流。
- AIter已有M1 `fused_allreduce_mhc_post`末尾缺少普通one-stage AR现有的
  `end_sync<ngpus,true>`，因此直接复用存在peer input覆写竞态。M32输入为256KiB，正常
  TP8 AR已走two-stage；不能简单批化one-stage。更合理的后续oracle是保留two-stage
  reduce-scatter及BF16 rounding顺序，只把MHC post/RMS epilogue并入stage-2 all-gather，
  先要求8-rank slowest latency每boundary至少节省8us、理想20us，再考虑接模型。
- 上述M32 two-stage epilogue随后已用隔离AIter `.so`完成8-rank graph oracle，正式
  `module_custom_all_reduce.so`未被覆盖。A为现有two-stage custom AR后接Triton
  MHC-post/RMS，B为stage-2 gather时直接执行四通道post并写RMS partial；final peer
  barrier包含在B计时内。1000-replay rank-max ABBA为A1/B1/B2/A2=
  `37.644/38.500/38.499/37.558 us`，A/B median=`37.601/38.499 us`，融合反而慢2.33%。
  8次变异输入没有stale-read或hang、八rank hash一致，但相对reference初始max-abs
  `2.44e-4`、变异最大`9.77e-4`，RMS sum最大relative差`5.45e-7`，不满足bitwise。
  因此未接SGLang生产路径；后续不再投入该epilogue，除非能同时消除final barrier成本
  并严格复现Triton reduction顺序。
- 继续验证了不增加第二compute launch的paired-A4单kernel：同expert连续A4 block映射
  到paired waves（gate/up）或paired 8-lane subgroups（TP8 K256 down），每个执行单元
  仍只保留A4 accumulator。recorder pass42/layer20严格重构32x6无重复路由，active=40、
  max occupancy=16；理论weight scans由64降到45（-29.69%）。但7轮ABBA实测gate
  `106.508 -> 134.228 us`、down `87.160 -> 113.296 us`、full stage
  `195.816 -> 249.724 us`，分别退化26.0%/30.0%/27.5%，虽三段均逐元素bitwise exact。
  leader/run判定与任务映射破坏了原grid-stride负载/缓存效率，权重复用没有转化为吞吐；
  独立原型撤回。当前A4 wave64已接近仅靠sort/block重排可获得的局部上限，后续需要改变
  FP4执行表示、减少量化/中间态，或做更大结构的RS-sharded residual，而非继续重排扫描。

### TP8 M32 hidden-RS / sharded-MHC结构oracle（2026-08-27）

- 新增完全独立、默认不接模型selector的
  `scripts/rocm/bench_dsv4_tp8_rs_mhc_oracle.py`。attention边界候选保持MHC residual
  按H分片：row-parallel partial `[32,4096] bf16`先RS成每rank `[32,512]`，本地执行
  MHC post；MHC pre先AR `[32,25] fp32`（24个mix dot partial加residual sumsq）。pre
  weighted sum按production落BF16，再AR `[32,1] fp32`取得全H sumsq，严格在RMSNorm后
  再落一次BF16；之后以K=512做row-sharded replicated projections并一次BF16 AR输出。
  没有把inverse-RMS穿过BF16 rounding，避免为省一个collective而改变production数值语义。
- 现有AIter
  RS只scatter dim0，普通row-major `[M,H]`必须先重排成`[P,M,H/P]`；正式kernel需要让
  producer直接写rank-major layout，或实现last-dim hidden RS，不能把额外transpose藏在
  micro之外。
- 八rank strict-semantics结构容差通过（不是模型correctness通过）：RS后的MHC-post shard相对full-H reference
  `max_abs=0`；plain wqkv N1536的sharded layer-input/final projection `max_abs`分别为
  `0.015625/0.015625`，projection relative-L2=`0.00258147`。C128将所有可复制projection
  打包为N2560（wqkv1536+core compressor1024）后，final `max_abs=0.015625`、relative-L2=
  `0.00260277`。差异来自K-shard GEMM/collective accumulation顺序，尚不满足bitwise；接
  真实权重前仍须layer20 tensor及France/long-context hash。三种projection shape的
  layer-input relative-L2均`1.71642e-4`、cosine=`0.999999881`，projection在八rank的
  output hash均只有一个唯一值；但N1536/N2560 projection relative-L2分别`0.00258147/
  0.00260277`，已超过初筛`1e-3`，必须标红为后续numerical blocker而非忽略。
- 五轮100-replay slowest-rank synthetic graph：plain N1536 full-H reference/candidate为
  `243.393/142.465 us`（-41.47%），C128 N2560为`247.498/148.283 us`（-40.09%）。这些
  PyTorch pointwise/GEMM数字只能作为继续写专核的promising oracle，不能当端到端收益。
  N1536 collective floor为RS=`18.342 us`、AR25=`15.061 us`、AR1=`12.755 us`、
  AR1536-BF16=`19.944 us`，四次通信合计约`66.1 us`；C128对应为`18.018/14.403/
  12.247/24.210 us`，合计约`68.9 us`。
- 早先BF16 AR1536=`19.9 us`与独立BF16 AR1537=`27.9 us`的差异已解释：1536 shape的
  byte count能整除TP8 vector pack，走vector two-stage；1537余64 bytes而进入
  `cross_device_reduce_2stage_naive`，不是计时矛盾。打包projection N必须保持TP8 pack
  对齐。
- C4把wqkv1536、core compressor2048、index compressor512和weights64合成N4160后，
  strict结构容差同样通过但不满足初筛：layer-input `max_abs=0.015625`、relative-L2=
  `1.71642e-4`、cosine=`0.999999881`；projection `max_abs=0.015625`、relative-L2=
  `0.00260659`、cosine=`0.999996543`，八rank hash一致。五轮100-replay full/candidate为
  `266.724/169.489 us`（synthetic -36.46%）；collective为RS=`18.475`、AR25=`14.499`、
  AR1=`12.389`、AR4160=`32.805 us`，合计`78.17 us`。FFN边界则是RS+AR25+AR1后，row-sharded router
  AR256与给expert gate/up恢复full-H的AG4096并行；只有attention边界在真实layer20上
  slowest-rank净省至少20us并1000 replay稳定后再实现，避免同时改两处跨层状态。
- 独立`bench_dsv4_tp8_collective_budget.py`以500 replay、7轮、逐轮8-rank max补齐严格
  通信预算并完成逐元素exact校验：AIter RS `[32,4096]bf16->[32,512]`=`17.540 us`，
  AR `[32,25]fp32`=`14.068 us`，AR `[32,1]fp32`=`12.193 us`，AR
  `[32,1536]bf16`=`19.659 us`；因此未融合的严格链为`63.46 us/layer`。作为替代布局，
  AR `[32,2560]bf16`=`23.844 us`、AR `[32,4160]bf16`=`33.622 us`，当前full-H
  AR `[32,4096]bf16`复测=`32.349 us`。同一micro的RCCL细粒度graph约
  `217--255 us/op`，不适合逐层方案；生产原型必须基于AIter peer-read。
- `AR1536=19.66 us`与此前`AR1537=27.16--27.87 us`的差异已定位为确定的layout
  fast-path边界，而非计时噪声。AIter two-stage在消息字节数可整除`world_size*16=128`
  时选vector kernel，否则选`cross_device_reduce_2stage_naive`：M32x1536 BF16为
  `98,304 B`且整除128，M32x1537为`98,368 B`且余64。二者都超过TP8的80KiB one-stage
  阈值。故不能用同质`1537xbf16`模拟真实`1536xbf16+1xfp32`；异构专核应保持1536 bulk
  的vector访问，在同一rendezvous附带FP32 scalar（或至少padding通信layout），否则会
  白白损失约7.5us。
- C4 N4160进一步对row-sharded projection partial做了BF16/FP16/FP32 A/B，并用
  full-H reference layer-input的本rank slice替换sharded-MHC输出做误差隔离。BF16
  normal/isolation projection relative-L2=`0.00260659/0.00260309`，FP16为
  `0.00149932/0.00149408`，说明两者误差几乎全部来自K512局部GEMM partial rounding与
  rank reduction，而非前面的sharded MHC；FP16虽同为2-byte通信且AR4160约`33.1 us`，
  仍未达到`1e-3`初筛。FP32 normal/isolation则降到`2.63233e-4/5.14203e-5`，数值过筛；
  isolation max-abs=`0.0078125`、cosine=`1.0`。所有variant八rank output hash一致。
- 性能上BF16 normal/isolation candidate为`169.489/168.182 us`；FP16为
  `259.259/259.034 us`，当前Torch/HIP FP16 local GEMM抵消全部结构收益；FP32为
  `260.651/262.847 us`，且独立500-replay AR4160-FP32保守值=`84.939 us`（本脚本短测
  `65.915 us`）。因此FP16是同带宽但数值仍不达标且kernel慢，FP32数值可接受但通信/计算
  预算不成立；三者都不能直接接production。若继续，必须写能以FP32 accumulator保留
  partial精度、却用2-byte或融合peer reduction避免FP32 bulk通信的专核，并在真实layer20
  FP8/BF16 cache权重上重新评估，而不是复用当前通用Torch GEMM。
- `/tmp/aiter_fp16_rs_projection_tuned.csv`的hipBLASLt定解进一步接入独立oracle（仅
  `--tuned-fp16`，未碰production）：N1536/2560/4160分别映射solution
  `12183/12008/11948`。必须通过`tuned_gemm.hipb_gemm`调用以创建每进程hipBLASLt
  extension；直接调用raw `hipb_mm`会让八rank全部native SIGSEGV，该失败路径已修正。
  C4 N4160 solution11948的五轮100-replay rank-max candidate=`220.138 us`，相对同轮
  full-H synthetic reference=`267.045 us`为-17.57%，也比通用Torch FP16 candidate
  `259.259 us`快约15.1%；collective仍为RS=`18.170`、AR25=`14.618`、AR1=`12.562`、
  AR4160-FP16=`33.117 us`。八rank output hash一致，projection max-abs=`0.015625`、
  relative-L2=`0.00149906`、cosine=`0.999998808`，与通用FP16数学基本相同，仍略高于
  `1e-3`初筛。因此定解证明FP16 compute可以回收约39us synthetic开销，但没有解决
  partial-rounding数值障碍，仍不能接production或宣称模型correctness。

### Layer20 M32真实projection tensor oracle（2026-08-27）

- 复用现有stage dump，仅增加默认关闭的layer/rank/row selector；固定11-ID France输入的
  position11、M32 decode在rank0保存了layer20 C4 `attn_norm=[32,4096] BF16`，以及四个
  production逻辑BF16矩阵：wqkv `[1536,4096]`、core compressor `[2048,4096]`、index
  compressor `[512,4096]`和index weights `[64,4096]`。独立
  `bench_dsv4_tp8_real_projection_oracle.py`让reference保持四次full-K BF16 `F.linear`
  各自舍入再concat；candidate才concat weight、按K512分片并做一次AIter AR。未接模型
  selector。
- 必须把static FP32 weight shard预先缓存。早先synthetic FP32 graph把约34MiB BF16 weight
  每replay转FP32，错误得到约260us；修正后graph只保留小activation cast。真实dump五轮
  200-replay rank-max：四projection reference=`123.536 us`，BF16 partial candidate=
  `40.347 us`（-67.34%），tuned FP16=`64.602 us`（-47.71%），FP32=`86.226 us`
  （-30.20%）。这些仍是projection子图、未包含跨层RS/MHC通信，不能直接折算E2E。
- 真实数值显著确认FP32 partial方向：BF16整体max-abs=`0.03125`、relative-L2=
  `0.0023567621`；FP16为`0.03125/0.0014017423`，仍未过`1e-3`；FP32为
  `0.0078125/4.5238427e-5`、cosine=`1.0`，八rank hash唯一。FP32分段relative-L2为
  wqkv=`3.91327e-5`、core=`5.75366e-5`、index-core=`8.26959e-7`、index-weight=`0`。
  因此真实权重没有复现synthetic中对FP32预算的悲观判断，下一步可进入teacher-forced/
  top-k-margin oracle，但仍不足以宣称生成correctness。
- 动态activation BF16→FP32 cast与预缓存FP32 activation的ABBA为A1/B1/B2/A2=
  `83.695/86.152/86.430/84.351 us`；cached-x反而慢约2.27us。小activation cast不是
  critical bottleneck，差异更像输入地址、cache状态或cast预热对后续GEMM的影响；不应从
  理想预算中武断扣掉cast时间。static weight cache才是此前FP32结果被严重低估的根因。
- 进一步要求hipBLASLt直接执行`BF16 input/weight -> FP16 output`时，M32/K512的
  N1536/2560/4160三形状均由`tuner/hipb_findallsols`返回0个solution；当前AIter
  mixed-output API不可用，raw `hipb_mm(solution=-1)`还会native SIGSEGV，禁止作为
  fallback。可运行的`BF16 GEMM -> FP16 cast`单GCD graph为
  `12.870/13.326/15.609 us`；常驻FP16 weight并每步将activation转FP16后调用上述
  tuned solutions为`12.523/16.388/29.157 us`，仅N1536快`0.347 us`（2.77%），后两者
  分别退化22.98%/86.80%。不含activation cast的理想FP16 GEMM为
  `9.360/11.734/22.001 us`。FP16路线相对FP32-accumulate再落FP16的relative-L2约
  `5.98e-5/7.14e-5/6.62e-5`，优于BF16输出再cast的约`1.67e-3`，但当前API与端到端
  延迟均不支持接production；该探针无代码改动、无checkpoint。
- Production consumer contract修正为mixed dtype：wqkv与index weights为BF16，两路
  compressor score tensor为FP32。Packed FP32 AR后只cast 1536/64两段；2048/512两段
  显式`contiguous()`保持FP32，四段均物化为storage-offset 0的连续tensor，避免split view
  的stride=4160泄漏给下游。CPU contract验证dtype/shape严格为
  `BF16/FP32/FP32/BF16`与`1536/2048/512/64`。
- 真实layer20 M32 dump计入四段物化后，FP32 candidate五轮200-replay rank-max=
  `100.716 us`，同轮reference=`121.409 us`，projection子图仍省17.04%；相对未计copy
  的约86.2us，连续化成本约14.5us。八rank hash一致。Mixed语义整体relative-L2=
  `0.00133577`，主要来自core/index-core score分别`0.00148367/0.00152420`；wqkv仍为
  `3.91327e-5`，index weight exact。当前reference compressor的FP32 tensor实际承载
  BF16 GEMM舍入值，而candidate保留真正FP32 partial，因此不能仅以“更高精度”为由接受；
  必须继续做compressor下游与teacher-forced/top-k margin验证。
- 默认关闭的TP8 FP32 attention-prep production实验随后完成端到端否证。仅C4开启的B服务
  hot三轮为`965.95/969.65/972.88 tok/s`，相邻A基线hot为`966.51/966.62 tok/s`，差异
  落在服务波动/噪声内；C4+C128扩展后的B2首轮cold=`852.79 tok/s`，hot仅
  `963.34/964.25 tok/s`，同样没有收益。两个candidate的full graph tiers France oracle
  均为`107/107`逐token一致，但projection micro的局部节省没有转化为E2E critical-path
  收益。因此production env、cache、loader post-quant hook和forward接线均已撤出；保留真实
  tensor oracle与默认关闭的stage dump，供后续结构研究，不能将该路线列为有效优化。

### TP8 routed-MoE结构探针与PP2并发实验（2026-08-27）

- 真实BS32 expert recorder仍可复用：`logical_count=[4096,43,256]`，pass 42对应buffer
  index 47；layer 20有40个active experts、最大occupancy 16、共192 assignments。现有
  gfx90a A4/R2/W8/B832/LDS FP4专核在该分布下约`195--196 us`，其中gate/up约
  `106.51 us`、down约`87.16 us`；均匀synthetic约`293 us`，说明真实router偏斜会
  显著改变微核预算，后续MoE候选必须同时报告真实分布与synthetic。
- gate/up后融合INT8 group-32 quant的HIP原型在production shape I256/A4上保证
  intermediate BF16、INT8量化值与scale逐元素bitwise exact，但reference为
  `98.8--99.4 us`时，候选在80/104/208/416 blocks分别约`161.87/132.08/133.48/
  133.29 us`，退化约`+63%/+34%/+35%/+34%`。原因是融合后每个wave承担更多串行任务和
  block级同步，省下一个launch不足以抵消并行度损失；原型已删除。
- token-major direct-final down在真实M32链路约`756.90 us`，对比grouped链路
  `293.45 us`退化`+157.9%`，且输出并非bitwise exact（max-abs 16）。跨expert直接按
  token消费会丢失grouped kernel的权重tile复用，因此不能用“直接写最终输出”替代当前
  expert-major down。
- 通用MFMA32对真实40个active experts仍需把每个expert pad到32 rows：104 blocks/
  split2约`1088.1 us`，104/208/416 blocks split4约`768.54/794.51/720.61 us`，均远慢于
  当前A4 sdot约`196.2 us`；relative-L2约`1.24e-4`但非bitwise。结论是BS32 routed
  shape仍属于大量小M专家，不能直接套用dense MFMA32 tile；当前A4 sdot映射接近该局部
  设计空间的最优点。
- PP2xTP4已通过通用启动参数接线验证。`pp-max-micro-batch-size=16`、无async depth时
  BS32约`748 tok/s`；改为micro-batch 8、async depth 2后，32x64 wall=`5.412 s`，仅
  `378.39 tok/s`。单France输出逐tokenexact但延迟`6.07 s`；group France 1/2/4 exact，
  更大组合测试在超时前未完成。PP stages确实有重叠，但逐token AR的跨stage依赖制造流水
  气泡，拆小M又降低每stage kernel效率。PP2保留为扩大KV容量的特例（理论约2.57M token
  cache），不能作为BS32冲击1500 tok/s的主路径；后续不再纯参数扫描PP。

### split-MoE M>1 grouped-down correctness修复与协议边界（2026-08-27）

- 补测旧`TP8/DP-attention2/attention-TP4/MoE-DP2/MoE-TP4` split fast path的
  M8/M16/M32。未修复时France为BS1 `1/1`，BS8 `2/8`、BS16 `1/16`、BS32 `0/32`
  exact；BS8已有6种输出，属于确定的kernel correctness错误，不能测速。
- 根因是M>1进入grouped FP4 down：split路径先用`topk_ids=-1`排除另一DP replica的
  slots，sorter因此不会为这些slots产生任务；但wrapper分配
  `partial=[M,topk,N] FP32`时使用`torch.empty`，final reduce却无条件读取并累加全部
  6 slots，把未初始化显存混入输出。BS1走slot-range direct kernel，没有该partial，
  所以旧BS1 oracle一直无法暴露此bug。
- 修复只在split grouped路径将partial显式清零；普通完整Top-6路径仍使用`empty`，不增加
  正式TP8/EP1开销。修复后full graph tiers `1/8/16/32`的France合计`57/57`逐token
  exact、每tier输出唯一。一次额外的跨MoE-DP Top-K broadcast没有改善未修复结果，已
  撤回，确认首故障不是router replica浮点分叉。
- 但256-token并发仍暴露更高层协议阻塞：两个独立DP scheduler在prefill/JIT和后续
  decode中形成不同local batch shape（日志可见同阶段M22/M33等差异），最终进入同一个
  TP8 collective时失配；请求与`/health`同时挂住。服务停止后八卡显存完全释放。因此
  本修复应作为M>1 grouped kernel correctness checkpoint保留，但split-MoE大并发仍不
  可交付，也没有有效吞吐数字。下一步若重访，必须由一个authoritative scheduler广播
  decode batch plan、request row order、active mask和graph tier，使两组逐step lockstep；
  仅让controller向两组发送相同请求不构成协议保证。
- 另一个独立的gate/up packed-FP4成对存储原型在真实pass47/layer20上逐元素bitwise
  exact，但完整stage由`195.836 -> 198.535 us`，退化1.38%；重排/地址依赖抵消合并VMEM
  load收益，原型已撤回，不接production。

### 当前TP8同服务BS8/16/32矩阵复核（2026-08-27）

- 在`d2c118d116`、单模型`TP8/EP1/no-A2A`、1M-token pool、SBO+ROCm multistream、
  A4/R2/B832 LDS unpack、TP8 K256 8-lane down下，只捕获`1/8/16/32`，固定同一组
  11-token France input IDs。France合计`57/57`逐token exact、每tier输出唯一。
- 256-token native AR同服务结果：BS8=`326.58/336.85 tok/s`，BS16=
  `629.60/629.93 tok/s`，BS32首两轮=`962.98/963.47 tok/s`。BS32随后六轮为
  `958.14/965.13/966.12/960.26/964.15/959.35`，median=`962.20`、trimmed=
  `962.22 tok/s`。所有请求均实际输出256 tokens。
- 因此当前跨服务可信运行包络应写作约`962--991 tok/s`，而不是只报告此前约990的
  高频态。BS8到BS16约1.90x，BS16到BS32仅约1.53x，效率损失主要集中在M32每层
  attention/routed-MoE tail。按本轮trimmed，BS32 step约`33.26 ms`；到1500目标的
  `21.33 ms`仍需净省约`11.93 ms/step`（35.9%，约277us/layer）。
- 256-token completion在BS8为2种hash、BS16为1种、BS32为既有7种hash；短France
  完全一致但长decode跨slot漂移债务仍存在。本轮没有生产代码改动，不能把短oracle扩大
  为长序列bitwise parity结论。

### TP8 BS32 BF16 hipBLASLt穷举与wqkv定向否证（2026-08-27）

- AIter hipBLASLt对八个真实M32 BF16 shape逐个穷举2226个solution。独立CUDA Graph
  五轮ABBA中，`wqkv_a [32,4096]x[1536,4096]`的solution 3931由
  `37.245 -> 28.958 us`（快28.62%，省8.29us）；core compressor N2048的solution
  3929由`40.709 -> 33.846 us`，N512的solution 5042由`25.162 -> 16.052 us`。
  其余主链候选没有价值：wq_b N8192/K1024慢6.26%，wo_b N4096/K2048仅快0.79%，
  shared-down N4096/K512慢2.06%。所有定解finite、12次graph replay bitwise稳定；相对
  Torch reference的relative-L2不高于`4.42e-5`。
- 只在wqkv_a、且仅M32调用点启用solution3931后，graph capture成功，France tiers
  1/8/16/32在A与B均为`57/57`逐tokenexact。服务级A/B/B/A的hot BS32结果为：A1约
  `942.1`，B1约`943.4`，B2约`943.1`，A2约`946.4 tok/s`。局部8.29us/layer节省没有
  转化为端到端收益，B相对末端A约退化0.3%；生产selector、layer标记和固定solution接线
  已全部撤回。
- 保留`scripts/rocm/bench_aiter_bf16_m32_gemm.py`作为独立复现工具。结论与此前N512
  “裸graph快42%、E2E慢约1%”一致：M32 dense GEMM的单stream局部时间不是当前graph
  critical tail，不能把hipBLASLt tuner结果直接全局接入。后续转向真实routed-MoE
  K-tiled packed-FP4 cooperative reuse；其继续门槛是完整routed stage从约195.8us降到
  不高于145us，并保持BF16逐元素exact。

### routed FP4 cooperative A8 LDS复用否证（2026-08-27）

- 按真实pass47/layer20路由实现过一个仅用于micro的gate/up原型：sorter直接用A8
  metadata；每CTA两条wave、每wave仍只保留A4 accumulator；两wave处理同一expert/row
  tile的前后四个routes，并由128线程把K2048 packed-FP4 tile和scale协作搬入LDS。lane
  仍按`g,g+64`累加，shuffle树不变，因此完整stage相对A4/R2/B832/LDS reference为BF16
  逐元素exact、max-abs=0。
- 但七轮50-iteration中reference median=`194.946 us`，cooperative A8 median=
  `546.881 us`（`+180.53%`）。A8把真实weight scans从约61.46降到约46（仅25.2%），
  完全不足以抵消global-to-LDS store、每tile block barrier、两wave LDS重读和有效并行度下降。
  因结果远高于145us继续门槛，C++ kernel、Python wrapper和benchmark接线均已撤回。
- 该结果与此前整行decoded-LDS staging失败一致：gfx90a routed小M当前需要保持无block
  barrier的wave级persistent任务。后续若重访weight reuse，应采用不要求同CTA wave同步的
  表示（例如离线layout/跨route寄存器映射），不能继续扩大LDS staging tile。

### TP8真实FFN边界hidden-RS oracle（2026-08-27）

- 扩展`bench_dsv4_tp8_rs_mhc_oracle.py`覆盖FFN边界：attention `wo_b` rank-local
  partial先reduce-scatter为H512 shard；MHC residual/post/pre保持hidden shard；router
  使用K512 row-sharded FP32 partial并AR256；同时把normalized H512 shard all-gather成
  routed/shared experts需要的完整H4096输入。当前oracle保守地串行router AR与expert AG，
  尚未把两支overlap收益计入。
- 随机结构输入中，BF16 router partial版本由`225.203 -> 171.753 us`（-23.73%），但
  router relative-L2=`2.616e-3`；FP32 partial把relative-L2降到`2.986e-4`，仍由
  `225.272 -> 180.153 us`（-20.03%）。FP32 primitive rank-max为RS4096=`17.44us`、
  stats AR25=`14.14us`、norm AR1=`11.82us`、router AR256=`23.43us`、expert AG4096=
  `34.39us`。
- 随后从真实TP8 eager层20、M32 decode抓取八rank `wo_b` partial、MHC state/参数、norm
  和router weight；所有partial FP32求和再落BF16与production `attn_out`逐元素exact。
  使用这些真实tensor时，reference/candidate为`225.847/181.851 us`（-19.48%）；sharded
  layer-input relative-L2=`1.147e-5`，FP32 router logits relative-L2=`1.424e-4`。
- 从raw checkpoint加载`layers.20.ffn.gate.bias`并执行真实sqrt-softplus+bias Top-6：
  32/32行expert集合完全一致，最小第6/7名margin=`4.9973e-4`，candidate最大choice-score
  扰动=`1.9455e-4`。debug dump接线的服务France tiers1/8/16/32也为`57/57`逐tokenexact。
  该结果达到进入默认关闭production原型的数值/预算门槛，但不能只在FFN边界临时分片：
  MHC residual必须跨层持续保持shard，MoE output和attention `wo_b`都应以RS结束；attention
  replicated projections改为K512 row-shard+AR，expert入口才AG full-H，最终层再恢复full-H。
- 另一个无barrier half-wave A8 gate原型（lane halves分别A4）在真实路由中为
  `106.310 -> 128.611 us`（+20.98%），且非bitwise（max-abs=`0.0625`、relative-L2=
  `4.10e-6`）；原型已撤回。当前不再投入A8 weight-reuse重排。

### TP8 K256 down half-wave A8权重复用否证（2026-08-27）

- 进一步隔离测试了down projection：每个wave划分四个16-lane pair，每个pair的低8 lane
  只加载一次K256 packed-FP4权重与scale，再用wave64 shuffle广播给高8 lane；两个half各自
  保持A4 accumulator和原有width-8 reduction，因此不引入block barrier或LDS staging。
- 在真实pass47/layer20、M32路由上，候选相对A4/R2/B832 LDS reference逐元素exact、
  `max-abs=0`，但七轮50-iteration median为`85.083 -> 128.903 us`（`+51.50%`）。
  wave shuffle、A8活跃状态和寄存器压力明显超过减半权重读取的收益。
- C++ kernel、Python wrapper和benchmark开关已全部撤回。结合gate/up half-wave与LDS A8
  两个反例，A8 route pairing不再作为当前TP8优化方向；继续推进跨层persistent hidden
  shard，它在真实层20 tensor oracle中仍有约19.5%的FFN边界收益。

### TP8 native HIP sharded-MHC真实张量oracle（2026-08-27）

- 新增了尚未接production、默认不可达的gfx90a H512 sharded-MHC三阶段HIP JIT：stage1
  完成BF16 hc_post并输出rank-local FP32 `[M,25]`（24 dots+sumsq）；AR25后stage2完成
  Sinkhorn与BF16 weighted residual并输出local y-sumsq；AR1后stage3完成本rank H512
  RMSNorm。`fn`按每个HC的同一hidden slice重排为连续`[24,2048]` FP16。
- 在真实TP8 eager layer20、M32 dump上，完整oracle边界包含RS4096、native sharded-MHC、
  FP32 router AR256和expert-input AG4096。严格A/B/B/A、每腿7轮×200 graph replay的
  rank-max合并median为reference=`225.929 us`、candidate=`116.961 us`，即`-48.23%`；
  两条candidate腿分别为`116.918/116.980 us`，收益不是热态顺序伪影。
- 数值检查：residual relative-L2=`2.261e-5`，layer-input relative-L2=`3.379e-4`，
  router-logit relative-L2=`3.506e-4`；八rank candidate hash唯一。加载raw checkpoint
  layer20 gate bias并执行真实choice math后，32/32行Top-6 expert集合完全一致，最小6/7
  margin=`4.9973e-4`，candidate choice max-abs=`4.52995e-4`。
- 该结果通过进入默认关闭的43层persistent hidden-shard bring-up门槛，但尚不能外推为E2E
  收益或生成correctness。production必须整次forward严格gate为gfx90a/TP8/EP1/no-A2A/
  PP1/decode，并依次验证单层partial、43层eager France、全graph tier France、固定
  continuation teacher-forced logits和1000 replay稳定性，之后才允许测速。

### TP8 persistent hidden-shard Phase1完整模型bring-up（2026-08-27）

- 新增默认关闭的`SGLANG_DSV4_GFX90A_TP8_HIDDEN_SHARD`整次forward分支，只在
  gfx90a/TP8/attnTP8/moeTP8/EP1/no-A2A/PP1/attnDP1/native decode全部满足时进入。
  层间状态固定为H512 shard；attention `wo_b`和TP-sharded routed+shared MoE输出以
  last-dim reduce-scatter结束。Phase1仍在attention与expert入口各AG一次，目的是先验证
  43层状态/collective协议，不作为最终性能设计。
- 补齐首层pre-only native HIP stats kernel。真实layer20输入相对full-H FP16-fn reference
  的global stats max-abs=`9.765625e-4`、relative-L2=`5.891e-8`。完整43层eager固定
  France oracle逐tokenexact，输出`[671,6102,294,8760,344,2619,51119,42499,1]`。
- 首次M32 graph capture暴露`reduce_scatter_along_dim(dim=-1)`返回movedim非连续view；
  M1 eager因退化shape掩盖该问题。在attention/MoE RS出口显式contiguous后，tiers
  `1/8/16/32`全部capture成功；固定France合计`57/57`逐tokenexact且每tier唯一。
- 未开启`SGLANG_DP_USE_REDUCE_SCATTER=1`时，RS回退通用collective，当前8192-token
  小池的32-request×128-token初筛仅`139.01 tok/s`。开启AIter custom RS并重新通过
  tiers correctness后为`234.17/235.92 tok/s`；从per-request wall可见小池只同时admit
  约16请求，所以这不是正式BS32，但即使按BS16也远低于既有约630 tok/s。
- Phase1性能失败不否定native sharded-MHC本身的117us oracle；它说明每层两次AG、两次
  RS、四次小AR及当前串行顺序远超局部MHC节省。内置stage profiler与custom-collective
  graph组合还会卡在ROCm queue-interposition signal，只产出EXTEND trace，故不采用其
  失真数据。下一步须直接进入Phase2：移除attention前AG，把replicated attention prepare
  projections改为K512 row-shard+output AR；在此之前Phase1保持默认关闭且不得报告为收益。

### TP8 persistent hidden-shard Phase2/3与collective隔离（2026-08-27）

- Phase2以默认关闭的`SGLANG_DSV4_GFX90A_TP8_HIDDEN_SHARD_ATTN_ROW=1`移除了
  attention入口H4096 AG。每层缓存K512 FP16 row shards，把qkv/core-compressor/
  index-compressor/index-weight四组投影按实际层拼成N1536/2560/4160 hipBLASLt GEMM，
  分别固定solution 12183/12008/11948，随后只做一次拼接输出AR并把bundle传给既有
  attention消费者。完整graph tiers1/8/16/32 capture成功，France合计57/57逐tokenexact。
- 32请求端到端结果仍为负：128-token为`604.7/627.8/627.3 tok/s`，256-token早期单轮
  为`642.33 tok/s`；独立返程服务256-token达到`653.99/655.22/655.37/655.00 tok/s`。
  即使采用较高稳定状态，也比正式TP8/EP1约962 tok/s基线慢约32%。恢复attention
  multi-stream prepare hook对结果无可测收益，说明损失不是该hook遗漏。
- Phase3曾把learned router改成K512 FP32 local GEMM+AR256，再进入expert-input AG。
  该路径保持tiers1/8/16/32 France 57/57 exact，但128-token热态仅
  `616.77--617.25 tok/s`，低于Phase2；串行增加一次全局rendezvous不值得保留。
  现由`SGLANG_DSV4_GFX90A_TP8_HIDDEN_SHARD_ROUTER`单独默认关闭，不能随ATTN_ROW隐式启用。
- 为隔离RS本身，增加默认关闭的ATTN/MOE `AR+local slice`诊断开关。两处同时切换后
  France 57/57 exact、128-token热态`619.46--619.76 tok/s`；attention-only在BS1/32
  France 33/33 exact、256-token热态`645.38/646.50/646.45 tok/s`，反而比返程RS/RS
  的约655 tok/s慢1.4%；MoE端切换近似中性。因此AIter RS并非结构退化主因。
- 预算模型显示candidate每层约9次collective：两个MHC各AR25+AR1、attention projection
  AR、wo_b RS、router AR（仅Phase3）、expert-input AG、MoE-output RS；baseline约两个
  H4096 AR。裸collective增量只解释约5ms/token，剩余约12ms主要符合多次rendezvous反复
  放大rank arrival skew的现象。persistent hidden-shard现作为correctness-valid oracle保留，
  不进入正式性能配置；主线返回约962 tok/s的full-hidden TP8 graph。
- 同期审计确认`SGLANG_USE_AITER_AG=1`在本机AIter版本实际静默无效：SGLang要求新版
  `should_custom_ag`与带`dim`参数API，本机0.1.11.dev32+g9a469a608均缺失。更重要的是
  当前`module_custom_all_reduce.so`早于源码新增AG end-sync barrier，不能直接绕过selector；
  接线前必须重编该module，并以变异输入做多轮graph replay八rank逐元素exact验证。

### 8-GCD pipeline-parallel对照结论（2026-08-27）

- 两个TP4 stage组成的PP2在BS32 native AR约`748 tok/s`，异步pipeline约`378 tok/s`，
  均低于单副本TP8/EP1约`962 tok/s`。PP可通过按层切分权重增加KV容量，但decode每token
  都要跨stage且BS32不足以摊平bubble；它不是当前1500 tok/s吞吐目标的可用主线。
- 8 GCD拆成两个独立TP4副本可获得较高聚合吞吐，但重复权重导致KV容量近似四卡配置，
  只保留为短请求特例。需要长上下文和大并发时，正式结构必须是单副本TP8；PP不能冒充
  TP8容量/吞吐结果。

### M32 dense INT8-weight/BF16-input A-tile原型（2026-08-27，未接正式路径）

- 为`[32,4096] @ [1536,4096]^T`实现了独立两阶段gfx90a HIP micro：第一核按
  token row做per-row symmetric INT8 activation quant，第二核由wave64 `sdot4`
  消费per-output-row symmetric INT8 weight cache。每个wave在A4或A8 token tile内
  只加载一次weight vector；module与benchmark没有接入production selector。
- 真实layer20 attention-normalized input与`wqkv_a`权重上，A4 candidate相对同一
  dequantized INT8 weight cache的BF16 `F.linear`为`35.423 -> 38.156 us`（`+7.71%`）；
  A8因寄存器/occupancy压力退化至`52.285 us`（`+48.85%`）。两者重复执行均bitwise
  稳定。synthetic A4十轮trimmed ABBA为`35.283 -> 38.302 us`（`+8.56%`）。
- 真实tensor的candidate `max-abs=0.0610352`、relative-L2=`0.111968`；用纯Torch
  对相同activation做per-row INT8量化得到relative-L2=`0.111944`，确认大误差来自
  activation quant而不是HIP实现。synthetic随机输入误差较小（relative-L2=
  `0.009449`），不足以代表真实DSV4分布。
- 结论：A4/A8跨token weight reuse仍不足以抵消activation quant launch与sdot4路径，
  且per-row A8量化不满足正式correctness门槛。该原型只保留为lower-bound micro，
  不应接入M32 attention projection或服务启动脚本。

### TP8 M32 routed/shared finalize融合oracle（2026-08-27）

- 新增隔离脚本`scripts/rocm/bench_dsv4_tp8_tile_finalize_oracle.py`，固定真实布局
  `[M=32,T=6,N=4096]`。严格slot树为`((p0+p4)+(p1+p5))+p2+p3`，先舍入BF16，
  再与shared BF16相加并再次舍入，最后调用现有TP8 custom all-reduce。实验只加载
  `module_custom_all_reduce_mhc_m32.so`，没有覆盖正式`module_custom_all_reduce.so`，
  也未接production selector。
- 单launch中“本地finalize写registered scratch -> system-scope CTA barrier -> peer-read AR”
  的原型不可用：本地scratch与reference全元素exact，但跨rank结果稳定有约7/8元素
  不一致；增加system fence、分别模拟owner-rotation与固定rank顺序均不能修复。该协议
  在现有RankData/Signal实现上不得接入production，需要独立的两阶段publication协议。
- 安全fallback只融合本地严格slot reduce与shared add，继续复用现有custom AR。1000次
  变异输入graph replay全部逐元素BF16 exact，首尾八rank SHA256一致，无stale/hang。
  7轮ABBA、每letter 200 replay的rank-max中位：A三launch `40.561 us`，B两launch
  `39.146 us`，收益`3.49%`（14个A与14个B样本）。这是每层约`1.42 us`的小收益上限，
  未达到5%提交/production接线门槛，保留为独立oracle数据。

### TP8 BS32 scheduler轮询间隔否证（2026-08-27）

- 固定单模型TP8/EP1/no-A2A、1M pool、graph tiers1/32和11-token France IDs，仅把
  `scheduler_recv_interval`从1改为4。B服务France在BS1/32为`33/33`逐tokenexact；
  32请求各256-token六轮为`952.08/962.44/957.01/955.07/962.42/958.31 tok/s`，
  中位约`957.66 tok/s`。
- 回程A恢复interval=1后France仍`33/33` exact，六轮为
  `989.88/969.01/969.57/968.88/969.41/965.29 tok/s`，去掉高态后中心约
  `968--969 tok/s`。interval=4约退化1.1%，说明固定BS32的CPU request polling不是
  当前critical bottleneck；增大间隔反而增加admission/response抖动。临时脚本入口已撤回。
- `num_continuous_decode_steps`在当前仓库只有ServerArgs声明，代码中没有实际消费者；
  在实现并证明每一步仍是单token AR、且逐step正确更新采样/cache之前，不能把该空参数
  当作可用优化开关。

### split-MoE authoritative scheduler lockstep修复（2026-08-27）

- `TP8/attention-DP2/attn-TP4/MoE-DP2/MoE-TP4`先前M>1并发挂死的直接日志证据是
  同一步DP1形成prefill M22、DP0形成M33，随后共同进入TP8 collective而shape失配。
  split fast path让`require_mlp_sync()`返回False，同时controller把generate分别发给
  两个DP leader；两边独立nonblocking ZMQ drain/admission，因此相同请求集合不保证
  同一poll、row order或batch tier。`dp_attention_local_control_broadcast`只同步control
  fanout，不广播work plan，不能解决该问题。
- 默认关闭的split env下，controller现只把generate发给authoritative DP0 scheduler；
  TP rank0随后在完整TP8 CPU group广播同一ordered work/control list。非authoritative
  DP socket若仍收到work会fail-loud，避免再次以设备端collective自旋表现。普通DP和正式
  TP8/EP1路径完全不变。
- 修复后graph tiers1/8/16/32全部捕获，France合计`57/57`逐tokenexact且每tier输出唯一；
  32请求各256-token连续五轮全部完成，证明原M22/M33 hang已解除。性能为
  `638.31/659.85/659.01/659.38/659.25 tok/s`，热态约`659.4 tok/s`，远低于正式
  full-hidden TP8约`965--969 tok/s`。因此这是应保留的同步correctness/fail-loud修复，
  不是1500 tok/s性能checkpoint；复制attention与每poll full-TP CPU rendezvous的成本
  使该结构不再作为当前吞吐主线。
- 为隔离CPU rendezvous，把split-only full-TP request broadcast改为device group、
  `force_cpu_device=False`。France tiers仍`57/57` exact，五轮BS32为
  `662.54/662.73/662.82/662.74/647.83 tok/s`；相对安全Gloo lockstep约659.4仅
  `+0.5%`且仍有慢态，远不能解释与正式TP8约969的差距。device broadcast已撤回，
  保持已提交的CPU-group correctness路径。

### TP8 BS32 full-K FP16 projection下界（2026-08-27，未接正式路径）

- 使用真实layer20 attention-normalized输入`[32,4096]`与`wqkv_a`权重
  `[1536,4096]`，比较ROCm BLAS BF16 `F.linear`与输入/权重均转FP16、输出再转
  BF16的完整投影。该实验不修改production selector，也不改变服务配置。
- 数值差异为`max-abs=0.00390625`、relative-L2=`0.00145310`、cosine=
  `0.999998927`；虽然明显好于per-row INT8 activation quant，但仍不是逐元素exact，
  若接入正式路径必须重新做逐token France与长序列correctness。
- 多轮ABBA中位为BF16 `38.728 us`、FP16加输出cast `42.388 us`，FP16候选慢
  `8.64%`。因此gfx90a/当前ROCm BLAS下，FP16并没有提供可利用的full-K projection
  快路，额外cast还扩大了开销；不应接入正式路径。

### TP8 BS64批量扩展探针（2026-08-27）

- 固定单模型`TP8/EP1/no-A2A`、1M token pool与France 11-ID输入，只捕获graph
  tiers `1/32/64`；为tier64把静态MXFP4 quant row上限提高到384。该实验没有改权重、
  kernel数学或speculative配置。
- 9-token France oracle为BS1=`1/1`、BS32=`32/32`、BS64=`64/64`逐tokenexact，且每个
  tier只有一个输出hash，证明新增tier64 capture/cached-decode路径正确。
- 同服务32请求各256-token四轮为`955.73/959.70/964.75/954.82 tok/s`；64请求四轮为
  `1324.99/1327.94/1326.93/1299.71 tok/s`，前三轮稳定中心约`1326.6 tok/s`。每个请求
  都实际输出256 tokens；长completion仍出现仓库已记录的跨slot greedy hash漂移，
  没有出现新的长度或完成错误。
- 从BS32翻倍到BS64只增加约38% aggregate throughput，且step从约33.3ms增加到约
  48.2ms；因此1500 tok/s不是简单提高并发即可达到。BS64可作为容量/吞吐特例，但BS32
  目标仍需缩短attention与MoE critical path。探针服务测试后已停止。

### TP8 C4 attention projection output-N分片oracle（2026-08-27）

- 新增独立脚本`scripts/rocm/bench_dsv4_tp8_output_n_projection_ag.py`，不接production
  selector。A使用真实layer20 M32输入分别执行wqkv-a/core-compressor/
  index-compressor/index-weight四个BF16 GEMM（总N=4160）；B把全局输出N连续切成
  8个N520 shard，每rank只做一个full-K GEMM，再用AIter registered all-gather和
  graph内rank-major重排恢复`[32,4160]`。
- AIter AG协议本身通过：8 rank的local GEMM结果与gathered对应slice均逐元素exact，
  所有rank得到相同A/B hash。7轮rank-max ABBA复测为A=`123.858 us`、B=`64.491 us`，
  局部减少`47.93%`；另一独立轮为`123.092 -> 63.897 us`（`-48.09%`）。这满足继续
  production A/B的性能门槛。
- B尚不满足中间tensor bitwise：四个原shape GEMM与合并后N520 GEMM选择不同BLAS归约，
  1000/1000个变异输入存在差异，最大absolute=`0.03125`、relative-L2=
  `1.126e-4`；初始差异主要在core-compressor（`0.0078125`）。通信和重排不是误差来源。
- 保留四个投影边界、分别计算N192/N256/N64/N8 shard可让四段逐元素exact，但四个小GEMM
  本地即`174.52 us`，已慢于A约`118.01 us`，加AG后更无收益。因此不能用多小GEMM换
  bitwise；下一步应把单bundle方案做成默认关闭的端到端实验，以France、固定长token轨迹、
  compressor cache/state和跨tier重复性作为correctness gate。通过前不得设为正式默认。

### TP8 output-N production A/B与回退结论（2026-08-27）

- 为验证oracle上限，曾接入默认关闭、只命中gfx90a TP8 M32 native decode的临时
  production路径：C4使用N4160/TP8=N520，C128使用N2560/TP8=N320；本rankfull-K
  BF16 GEMM后由AIter registered AG恢复原tensor，并复用qkv/core/index现有precomputed
  hooks。M1、prefill、dense层和其他拓扑全部保持旧路径。
- 本机AIter已有正确的registered AG kernel，但旧Python类缺`should_custom_ag`且AG方法
  不接受`dim`，导致SGLang此前静默回退RCCL。新增SGLang dispatch adapter，仅为旧类补
  shape/alignment/workspace gate和`dim=0`接口；外部dirty AIter未修改。真实服务完成
  tiers1/32 graph capture，证明capture warmup、graph buffer注册和replay协议可用。
- C4+C128 B服务France连续三轮BS1/BS32全部逐tokenexact。BS32×256六轮为
  `995.32/1004.55/1004.29/992.53/1004.11/1002.52 tok/s`；去掉一次BF16中间copy后为
  `1000.58/1002.22/1003.25/1000.89/1003.22/994.94 tok/s`，没有进一步提高。
  同机A回程为`962.88/966.31/967.89/970.60/970.65/969.89 tok/s`，稳定中心约
  `969--971`；因此真实端到端收益只有约`3.3--3.6%`，远低于独立projection oracle的
  约48%。
- C4-only六轮为`986.58/992.62/987.95/990.04/991.16/992.32 tok/s`，仅约+2.2%；
  C128确有增益，不能靠缩小作用面同时保留全部速度。B的256-token主hash与A有重叠，
  France也正确，但次要greedy分叉集合改变；combined-N GEMM又已知1000/1000中间tensor
  非bitwise。它既未达到5%收益门槛，也未达到长轨迹严格parity门槛。
- 因此production模型接线和实验env已完整撤回，不留默认关闭的死分支；保留独立oracle、
  本节数据和通用旧AIter AG兼容adapter。若未来有能保持原GEMM归约语义的单launch专核，
  可复用oracle，但不应重新接当前combined-N hipBLAS方案。

### TP8 segmented grouped-MM + HIP pack oracle（2026-08-27）

- 为避免combined-N跨投影改变shape，扩展output-N oracle使用ROCm Torch原生
  `torch._grouped_mm`。每rank取全局连续N520 slice，并按原projection边界切成三个
  不跨边界、最大N256的group；八rank布局分别为`256/256/8`、`256/240/24`、
  `256/208/56`或`256/200/64`。输入用zero-stride expand复用同一`[32,4096]`，
  不产生activation复制。
- 新增独立HIP pack kernel，将`[3,32,256]`的valid rows按rank布局打包为连续
  `[32,520]`，然后复用AIter registered AG。pack对四种布局均逐元素exact。单GCD
  grouped GEMM本身约`32.9--33.1 us`；相比四个local小GEMM约`167.9 us`显著更快。
- 8-rank graph oracle的7轮ABBA rank-max为A=`123.951 us`、B=`75.818 us`，局部减少
  `38.83%`。真实未扰动layer20输入的完整N4160输出、四个segment、local AG slice及
  八rank hash全部exact，说明连续rank slice、pack和AG顺序正确。
- 但1000个确定性输入扰动中仍有812次与四个原shape F.linear不bitwise，最大absolute
  `0.03125`、最大relative-L2=`1.1033e-4`。误差来自`_grouped_mm`选择的MFMA归约，
  不是pack/AG；它只比combined-N的1000/1000分叉略好。预计端到端收益也低于已测
  combined-N约3.3--3.6%，不足以抵消长轨迹风险，因此只保留oracle，不接production。
### TP8 M64 grouped-FP4 A8几何否证（2026-08-27）

- 为解释BS64从约`1326.6 tok/s`继续扩展受限，先用现有generic grouped-FP4
  kernel在单GCD构造M64 oracle；没有修改production selector。M32 recorder推导的
  M64 surrogate中，A8相对A4理论可减少约`32.7--41.0%` weight scans，因此继续门槛
  预先设为full routed stage至少快`15%`。
- 固定随机种子、相同FP4 weight/scale与LDS LUT，A4/R2/W8/B832五轮中位为
  `488.484 us`（`487.668/488.004/488.484/489.772/491.132`）。A8扫R1、W4/W8与
  B832/1248/1664/2080时，最好是R1/W4/B1664的`483.340 us`，仅快约`1.05%`；
  其余R1约`490--505 us`，R2已退化到`519--527 us`。所有已完成几何输出均为finite，
  同assignment内逐元素exact。
- 结果远低于15%门槛，说明减少weight scans仍不足以抵消A8的寄存器/控制开销与有效
  并行度损失；停止剩余R2/R4 sweep，不启动完整服务、不修改selector。M64不能简单靠
  把当前A4静态改为A8得到5%收益。

### TP8 tile-epoch producer/reduce流水首轮smoke（2026-08-27，未接生产）

- 为修复旧single-launch tile-finalize约7/8元素错误，新增过隔离JIT oracle，使用独立
  produced/consumed/end epoch inbox、system-scope release/acquire、P/C双stream和严格
  rank0到rank7 FP32求和；没有修改production或外部AIter。JIT扩展可成功编译、加载并
  注册8-rank IPC buffer。
- 首次2-replay smoke在epoch handshake阶段出现八卡设备端同步自旋，已立即终止全部
  torchrun进程并用`amd-smi process`确认资源释放；因此尚无correctness或latency结果，
  不能宣称协议有效。下一步必须按eager P->C单轮、各自capture、首次replay三段加入
  host/device marker定位；在1000 replay exact前不得接真实`wo_b`或production selector。
- 分段eager已经把首故障定位清楚：producer完成后每个rank的`[16,8]` produced inbox
  只有自身rank一列为1，其余七列仍为0。也就是对普通AIter registered Torch tensor做
  peer system-scope atomic store没有形成可见写入，旧consumer因此必然自旋。后续协议
  改为每个rank只release-store本地epoch，等待方经`RankData`直接peer-load八个rank；
  consumed/end也采用同一local-store/peer-load模型。
- 更深的根因不是system-scope原语本身，而是本机AIter ROCm
  `CustomAllreduce._get_ipc_meta()`对普通Torch caching-allocator tensor取得IPC handle后
  把offset硬编码为0。子分配tensor的peer `RankData`因此可能指向allocation base而非
  tensor地址；改变allocation尺寸后会表现为wait自旋或epoch2混代。`comm.buffer`同样是
  caching tensor，不能作为offset oracle。改用`aiter.allocate_meta_buffer()`获得direct
  HIP allocation，并只注册一次base、以固定offset访问data/produced/consumed/end后，
  wait-only、load-only及完整eager/end均通过。
- 修正后的single-slot双stream graph依次通过2、10、100、1000 replay；每轮8 rank输出
  逐元素bitwise exact且rank hash唯一，epoch1000 SHA256前缀为`19276e78fe58`。这证明
  local release-store + peer system-acquire load + GLC payload load协议可稳定复用，不再
  有stale/mixed epoch。当前仍是独立oracle、没有接production；下一门槛是与“相同
  producer + 现有AIter AR”做slowest-rank ABBA，收益不足5%就不接真实`wo_b`。
- 公平性能门槛随后完成：A与B都包含相同synthetic producer、wait/ack/end和严格
  correctness；A使用现有AIter registered AR，B使用tile publication/reduce。7轮
  ABBA、每段200 replay、取8-rank slowest后，B为`83.107 us`，A为`65.480 us`，B反而
  慢`21.21%`；14个B样本仅`82.748--83.866 us`，14个A样本为
  `64.840--65.955 us`，AR-only为`27.037 us`。两路均8 rank exact、max-abs=0。
  因此不接真实`wo_b`；保留oracle用于复现IPC offset bug与system-scope协议，但不能将
  tile流水列为性能成果。

### CDNA2 packed-MXFP4 UDOT8 routed-MoE反例（2026-08-27）

- ISA审计确认gfx90a具有`V_DOT8_U32_U4`，Clang builtin为
  `__builtin_amdgcn_udot8(a,b,acc,false)`。E2M1 magnitude可用nibble-SWAR精确映射为
  `{0,1,2,3,4,6,8,12}`；符号分离后每8项使用
  `all=udot8(|x|,|w|)`、`opposite=udot8(|x|,|w|&sign_xor)`、
  `signed=all-2*opposite`。CPU对全部16x16 code pair（含正负零）及10万组随机8-lane
  向量均逐项exact。
- AIter HIP `per_1x32_f4_quant_hip`当前在gfx90a并非可用依赖：conda host hipcc先因缺
  `thrust/complex.h`失败；显式切到`/opt/rocm/core-7.14/bin/hipcc`后，`module_quant`
  又为gfx90a实例化gfx942/950专用`v_cvt_pk_fp8_f32`并被assembler拒绝。oracle因此仅用
  AIter纯位运算Triton MXFP4 quant作前端；核心gate/down仍为HIP UDOT8。这是AIter
  module级架构guard bug，不能被误写成UDOT8 kernel correctness问题。
- standalone `kPrepacked=3` oracle复用真实recorder pass47/layer20路由（M32、40个active
  experts、最大occupancy16）及A4/R2/W8/B832几何。量化后数学reference把MXFP4
  activation code无损展开成现DOT4所用signed INT8 code，并复用相同E8M0 group scale。
  UDOT8最终BF16相对该reference逐元素exact，`max_abs=0`、`relative-L2=0`，证明value
  nibble顺序、activation scale、weight shuffled-scale offset及符号公式全部正确。
- 但full routed stage两轮短smoke中，现LDS-DOT4为`203.202/201.346 us`、median
  `202.274 us`；MXFP4+UDOT8为`339.652/338.916 us`、median`339.284 us`，退化
  `67.7%`，远高于预设`<=125 us`继续门槛。SWAR、双unsigned dot和MXFP4 quant成本
  压倒省掉的LDS decode；不接production selector、不启动服务。该结果也说明不能只凭
  “DOT8每指令八元素”推断CDNA2 FP4收益，必须计入E2M1 sign-magnitude展开。

### TP8 occupancy bucket与attention consumer-fusion静态复核（2026-08-27）

- 现有768个完整BS32 pass仍来自同一prompt加不同salt，不能代表32条自然请求。按现dump，
  hash层0--2平均`52.62`个active experts，occupancy 1/2/3--4/5--8/9--16/17--32
  承载assignments为`5.96/8.83/44.16/24.77/9.03/7.24%`；learned层3--42平均
  `38.05`个active，分别为`3.53/5.49/26.87/23.18/20.93/20.00%`。
- 全43层A4平均`61.463` scans、`78.10%`利用率、padding `53.85` assignments/层。
  仅将run 1/2/3--4分成A1/A2/A4不会减少scan或weight bytes，只把padding降至约
  `21.56`；现kernel对invalid assignment已经continue，A1/A2块又只占约20.5%的A4
  scans，因此gross上限约4% routed stage（约8us/layer），新增launch后预计净0--2%，
  不能按35--70us/layer立项。
- run>4使用A8虽可把scan降到约`46.001`（-25.15%），但真实A8、sort8 light/heavy
  双核和paired-A4单核均已exact但端到端/完整stage退化。下一步只先采32条不同固定
  input IDs、至少128-token warm window并分hash/learned层输出直方图；理论scan下降
  不足20%立即停。若继续，独立bucket oracle必须把full stage压到不高于`166 us`
  （相对195--196us至少15%）且逐元素exact，才允许接selector。
- attention侧unified-KV已经只为local heads分配q_out，legacy才创建/zero 64-head
  q_padded；Q/indexer的RMS/RoPE/Hadamard/quant consumer fusion也已存在。dual
  compressor postprocess虽micro省23--25us/C4层，E2E仅+0.026%；HIP multistream
  ABBA约-0.13%。因此不能重复计算q_padded、dual或event收益。尚可做的局部oracle只有
  CK `wqkv_a + segmented RMSNorm`（真实目标约8--12us/layer）及M32 inverse-RoPE到
  wo_a（约8--15us/layer）；attention consumer fusion的40--70us净收益预算偏乐观。

### TP8 多样请求A1/A2/A4 occupancy-bucket实测（2026-08-27）

- 新多样请求dump `/tmp/expert_distribution_recorder_1787803355.1855972.pt` 有168个完整
  BS32 decode pass。丢弃前32个warm pass后，128-pass窗口中hash层平均`118.52`个active
  experts、A4 `122.98` scans；learned层平均`105.84`个active、A4 `112.65` scans。
  这与旧重复prompt仅约38--40个active experts的分布差异很大，也解释了多样请求下
  routed stage负担上升。
- 新增独立、默认不接production selector的
  `scripts/rocm/bench_dsv4_gfx90a_occupancy_bucket_oracle.py`。用代表性的pass37/layer34
  （active=106、max occupancy=7、A4 scans=113）严格重构32x6无重复top-k；CPU一次生成
  A1/A2/A4连续metadata（分别64/24/25 blocks）。三个gate producer共享`[M,T,I]`，三个
  down producer共享同一FP32 partial，最后只执行一次现有固定slot reduction，避免把
  reduction重复三次；CPU metadata构造约`1.826 ms`，未计入GPU stage，production GPU
  sorter只会增加额外成本。
- 五组合理block profile的gate BF16、down FP32 partial和最终BF16输出均逐元素exact、
  `max_abs=0`；最佳profile再对100次变异quantized input replay，最终输出也全部
  bitwise exact。7轮ABBA最佳bucket profile为gate `(624,208,208)`、down
  `(832,416,416)`：完整stage `294.270 us`，相同输入统一A4/R2/W8/B832为
  `257.434 us`，退化`14.31%`。分项为gate `154.491 vs 137.043 us`（+12.73%）、down
  `122.180 vs 110.337 us`（+10.73%）、quant约`39.429 us`、唯一reduce约`4.427 us`；
  其他bucket profile退化`15.87--26.00%`。
- 新分布虽然singleton很多，但A1/A2/A4不减少weight scans；三次launch与三次1KiB LDS
  LUT初始化的固定成本超过少量invalid-lane/VGPR节省。候选远未达到full-stage
  `<=166 us`或至少15%收益门槛，因此不接selector、不启动服务；若重访必须让单launch
  在不扩大常驻VGPR的前提下动态选择assignment width，不能继续做三bucket launch。

### TP8 down-consumer INT8 quant融合oracle（2026-08-27）

- M32两次generic Triton group32 INT8 quant的单GCD 7轮ABBA分别为：gate输入
  `[32,4096]`约`37.080 us`，gate输出`[32,6,256]`约`37.428 us`，standalone合计约
  `74.5 us`。已有HIP wave64专核分别为`16.393/16.749 us`，两shape的INT8值与FP32
  scale均相对Triton逐元素bitwise exact；但历史service A/B已表明单独换quant launch
  会被graph并行隐藏，不能据此推断约41us E2E收益。
- 现R2 gate epilogue无法直接量化：一个group32由16个独立wave task、可能跨CTA完成。
  历史R4/8-wave协作原型虽intermediate/INT8/final BF16 exact，scale最大差`7.9e-13`，
  但LDS与barrier让stage约`501 -> 620 us`。本轮改做独立down-consumer oracle：每个
  expert-block复制若干CTA；CTA从已舍入BF16 intermediate用32个16-lane subgroup生成
  A4x8组INT8/scale到LDS，随即循环自己的N-row shard，写原FP32 partial，最后复用同一
  fixed-slot reduction。没有production selector。
- 多样请求pass37/layer34（active106、A4 blocks113）下，CTA1/2/4明显退化；消除一次
  无依赖的block barrier后，扫描CTA4/6/7/8/9/10/11/12/13/14/15/16。7轮ABBA最佳
  CTA16为完整stage `257.536 -> 237.691 us`，节省`19.845 us`、`7.71%`；quant+down
  子链`115.458 -> 95.655 us`，节省`17.15%`。CTA11/12/13/14/15也分别省
  `6.60/6.12/5.82/6.08/6.57%`，说明收益不是单点噪声。
- CTA4--16所有测试点的FP32 partial和最终BF16均逐元素exact/max-abs0；CTA12和CTA16
  各通过100次变异输入，CTA16进一步通过1000次CUDA graph replay，每轮partial/output
  exact，无stale或hang。该局部积木超过5% micro门槛，但只省约20us/layer，尚低于当前
  E2E结构候选约39us/layer门槛；在wave-owned gate producer组合oracle达到`<=218 us`
  前不接service/production。

### TP8 wave-owned gate+SwiGLU+quant反例（2026-08-27）

- 为避免旧R4协作原型的跨wave/CTA group32同步，新建完全独立的wave-owned producer：
  每个A4 expert block固定2 CTA、每CTA 4 waves，每wave独占连续32个I rows；每行仍按
  原wave64顺序扫描K4096、使用相同shuffle归约、bounded SwiGLU并先舍入BF16。随后wave
  自己从LDS重读32个BF16值，严格复用HIP group32 quant的16-lane max/divide/cast次序，
  输出global INT8/scale给现A4 down与fixed reduction。没有production selector。
- 多样请求pass37/layer34中，intermediate BF16、INT8、FP32 scale、down FP32 partial和
  最终BF16全部逐元素bitwise exact，100次变异输入也exact。但7轮ABBA的gate+quant为
  `140.316 -> 395.165 us`（退化181.62%），完整stage `257.265 -> 513.167 us`
  （退化99.47%）。
- 旧重复prompt pass47/layer20（active40、max occupancy16、A4 blocks64）也通过相同全链
  exact和100次变异；gate+quant `109.306 -> 261.859 us`（退化139.57%），完整stage
  `195.401 -> 348.131 us`（退化78.16%）。根因是2 CTA x 4 waves把多样/重复分布的gate
  并行wave数压到约904/512，而每wave串行32行；省掉quant launch完全无法补偿并行度
  损失。候选远高于`<=218 us`组合门槛，停止该映射，不接service/production。

### TP8 MHC FFN-RMS quant producer与组合链门槛（2026-08-27）

- BS32正式路径已静态确认不是BS1-only native/fused tail，而是
  `mhc_post_combine_rms -> splitk pre-mix -> sinkhorn -> mhc_weighted_sum_triton ->
  _gfx90a_mhc_rmsnorm_kernel`。最后的RMS kernel是一token一program、H4096/8-warps，
  因此新增独立Triton oracle：保留原BF16 `ffn_input`给router/shared expert，并在同一
  program显式BF16 round-trip后reshape成128x32，输出routed gate所需INT8/FP32 scale。
  没有production接线。
- 使用真实rank0/layer20 residual与norm weight构造M32 weighted-sum输入，candidate的
  BF16 hidden、INT8和scale相对现Triton quant及HIP wave64 quant均逐元素bitwise exact、
  `max_scale=0`，100次变异也全部exact。7轮direct ABBA为现RMS+Triton quant
  `60.556 us`、RMS+HIP quant `50.900 us`、fused `23.900 us`，分别局部节省
  `36.656/27.000 us`，超过15us局部门槛。
- 随后按真实边界组合完整链：A=`weighted/RMS + Triton gate quant + A4 gate + Triton
  down quant + A4 down/reduce`；B=`weighted + fused RMS/quant + A4 gate + CTA16
  down-consumer/reduce`。多样pass37/layer34下hidden、gate q/scale、intermediate、FP32
  partial与final BF16全链bitwise exact；100变异与1000 CUDA graph replay也全部exact、
  无stale/hang。
- 但两个局部micro收益在完整链中不加和。两次正式7轮direct ABBA分别仅省
  `20.576 us`（`272.625 -> 252.049 us`, 7.55%）和`20.880 us`
  （`272.651 -> 251.772 us`, 7.66%）；同一已捕获graph的7轮ABBA为
  `267.443 -> 249.364 us`，只省`18.080 us`（6.76%）。这低于预设净省40us门槛，
  因此尽管局部producer很快，暂不增加AiterRunnerInput sidecar或production selector；
  后续若重访必须先证明graph内单项收益，而不能相加direct standalone数字。

- 为排除独立graph oracle低估service调度收益，曾用单一默认关闭开关临时接入严格
  `gfx90a + TP8/EP1/DP1 + no-A2A + native decode M32 + A4/R2/W8/B832/LDS`路径：MHC
  RMS producer经layer-key sidecar向AIter传gate q/scale，并以CTA16 down-consumer替换
  第二次quant+down。France全tier/候选合计`33/33`逐tokenexact。
- 正式service A/B仍未转化为可观收益：baseline中位`947.925 tok/s`，candidate正常轮
  中位约`958.7 tok/s`，仅约`+1.1%`，且candidate出现一次`873.4 tok/s`慢轮。该结果远低
  于5%提交门槛，并再次证明isolated kernel/graph节省会被完整scheduler与并行支路隐藏。
  production env、MHC可选输出、forward-batch sidecar、AIter透传及consumer便捷入口已逐
  hunk精确撤销；独立oracle header/wrapper/scripts与本实验记录保留。

### gfx90a M32 wqkv_a + segmented RMSNorm CK可行性否证（2026-08-27）

- 新增独立、默认不接production的
  `scripts/rocm/bench_dsv4_wqkv_segmented_rms_oracle.py`，复用真实layer20/rank0
  `attn_norm [32,4096]`、`wqkv_a [1536,4096]`及checkpoint中的q-norm weight。
  A严格保持production shape：单个N1536 BF16 projection后，仅对前1024维q-lora执行
  AIter RMSNorm；后512维为KV passthrough。B是对CK现有接口最乐观的lower bound：把
  projection拆成N1024/N512两个BLAS调用，再对q执行同一RMSNorm，尚未计入CK workspace、
  跨N-block finalize或额外layout成本。
- q raw、KV raw和normalized q三段在B中均相对A逐元素bitwise exact、`max_abs=0`，20次
  graph replay也保持bitwise稳定。7轮ABBA、每腿500 replay结果为A median
  `39.858 us`、B median `67.748 us`，拆分lower bound退化`41.17%`，未通过预设至少
  10%的继续门槛。
- component graph中原N1536 projection median为`36.762 us`，独立RMSNorm为
  `6.998 us`。即使把RMSNorm完全免费删除，full producer+consumer的理想收益也只有
  `8.42%`，仍低于门槛。CK现成单kernel `DeviceGemmLayerNorm_Xdl_CShuffle`又要求整个
  N由单workgroup覆盖（N<=NPerBlock）、实现标准LayerNorm而非RMSNorm，且不能表达
  “前1024 normalize、后512 passthrough”；跨N-block接口需要workspace和第二阶段。
  因此不再写CK segmented专核、不接服务；该方向的现实实现不可能超过已经不足10%的
  免费RMSNorm上限。

### gfx90a M32 inverse-RoPE→wo_a lower-bound否证（2026-08-27）

- 新增独立、默认不接production的
  `scripts/rocm/bench_dsv4_inverse_rope_woa_lower_bound.py`。真实layer20/TP8 rank0
  attention输出为BF16 contiguous `[32,8,512]`；TP8下仅一个local output group，
  所以`view [32,1,4096]`没有copy或layout transform。当前M32 `wo_a`为BF16
  `[32,1,4096] x [1,1024,4096] -> [32,1,1024]` einsum，不走仅支持M<=8的
  gfx90a wave64 grouped GEMV，也不需要FP8/quant输出布局。
- oracle从checkpoint读取layer20 FP8 `wo_a [8192,4096]`及E8M0 scale，严格按runtime
  128x128 block公式解量化为BF16并取rank0的1024-row shard。当前inverse-RoPE用模型
  compressed-YaRN config重建freqs。raw+inverse相对dumped inverse、重建weight的einsum
  相对dumped wo_a、完整A相对预旋转G均逐元素bitwise exact、`max_abs=0`；positions
  `0/4/127/128/512`重复执行也exact且finite，graph replay保持相同结果。
- A/G都计入相同的256KiB input copy以重置原位graph replay。7轮A/G/G/A、每腿500
  replay中，A=`copy raw + inverse + wo_a` median `37.312 us`，G=
  `copy pre-rotated + wo_a` median `35.030 us`；即使未来融合核完全免费消除inverse，
  理想也只省`2.282 us`、提升`6.12%`，远低于预设绝对至少`10 us`继续门槛。独立
  R=`copy+inverse`为`7.782 us`，C=`copy`为`7.036 us`，净rope kernel仅约
  `0.746 us`。
- 因此不实现CK/HIP MFMA A-load transform、不接服务。BS32已有正确选择：不把inverse
  epilogue塞进attention core，避免VGPR/occupancy损失；该局部consumer fusion的可删
  工作量本身不足，而不是缺少group-major或quant布局优化。

### TP8 BS32 SBO on/off配置复核（2026-08-27）

- 在同一commit、TP8/EP1/no-A2A、1M-token pool、仅捕获graph tiers 1/32且保持ROCm
  multistream开启的条件下，只切换`--enable-single-batch-overlap`。两边France BS1与
  BS32合计均为`33/33`逐tokenexact。
- SBO-on剔除首轮JIT后七轮为`947.842/948.687/948.853/946.821/947.925/948.714/
  946.260 tok/s`，median=`947.925`。SBO-off剔除首轮并单列一次`394.431`慢轮后，
  六轮为`943.773/943.900/943.976/947.869/943.381/943.501 tok/s`，median=
  `943.837`。SBO-on中心约快`0.43%`，关闭没有收益且稳定性更差。
- 代码审计仍确认当前AIter/StandardDispatcher的SBO并不会把M32拆成2xM16，脚本中
  “直接重叠shared/routed”的注释也不准确；但其周边调度/stream行为在完整profile中
  有微小正效应。因此保留现默认，不做配置清理，也不能把该0.43%写成主要优化成果。

### TP8 M32 MHC split-K geometry全扫（2026-08-27）

- 新增独立、默认不接production的
  `scripts/rocm/bench_dsv4_gfx90a_mhc_splitk_geometry.py`，使用真实rank0/layer20
  `residual [32,4,4096]`、FP32 `fn [24,16384]`、HC scale/base和FFN norm weight。
  stage0严格复用production `_gfx90a_mhc_mix_splitk_stage0_kernel`；fused tail只把原来
  硬编码的8-way reduction参数化，Sinkhorn、BF16 round-trip、weighted residual和
  RMSNorm数学顺序不变。
- 全扫`BLOCK_N=1/2/4/8`、`SPLITS=4/8/16`、`BLOCK_K=512/1024/2048`，每组7轮
  CUDA graph ABBA。production `N4/S8/K1024`稳定约`26.04 us`。最快的有效候选
  `N4/S16/K1024`为`24.906 us`，仅省`1.152 us/layer`（4.42%）；其post/comb因split
  reduction次序变化分别只有`8.94e-8/7.15e-7`最大误差，最终BF16 out仍bitwise exact。
  `N4/S8/K2048`约`25.035 us`，仅省`1.018 us`，最终out同样exact。两者都远低于
  `>=10 us/layer`门槛，不进入service A/B或production。
- `SPLITS=16/BLOCK_K=2048`是非法组合：每split的`CHUNK_K=1024`小于tile，stage0会跨
  split边界读取，产生约`1.62`的post最大误差和明显退化，不能误当geometry结果。
  `BLOCK_N=1/2`普遍更慢，说明M32下当前row并行度已经足够，继续扩大CTA数量没有收益。
  结论是M32 MHC pre-mix geometry已接近局部最优，attention/MHC的后续预算必须来自
  真正的producer-consumer中间态/同步消除，而不是继续扫split-K参数。

### TP8 diverse A1 wave-owned down oracle否证（2026-08-27）

- 新增完全独立、默认不接production的A1 short-run oracle：
  `gfx90a_fp4_a1_wave_owned_oracle.cuh/.py`及
  `scripts/rocm/bench_dsv4_gfx90a_a1_wave_owned_down_oracle.py`。保持TP8
  `K=256`的8-lane subgroup、group32向量load、SDOT4累加和shuffle reduction树不变；
  每个wave的8个subgroup共同处理同一个singleton expert的连续16个output rows，并把
  相邻wave映射到不同A1 expert。输出仍写原`[M,T,N]` FP32 slot，未改变最终fixed-order
  reduction。
- 使用diverse recorder pass37/layer34的真实路由分布：106个active experts、最大
  occupancy 7，其中64个A1 experts。blocks 104/208/416/832下candidate partial均相对
  现有A1 grouped kernel逐元素bitwise exact、`max_abs=0`；832 blocks再做100次变异
  INT8 input也全部exact。
- 7轮CUDA graph ABBA全部明显退化：104 blocks `72.590 -> 102.129 us`（+40.69%），
  208 blocks `58.141 -> 85.016 us`（+46.22%），416 blocks
  `53.173 -> 81.005 us`（+52.34%），832 blocks `52.139 -> 87.382 us`
  （+67.59%）。现kernel让相邻subgroup处理同expert的连续rows，天然保留连续weight
  stream/L2 locality；强制相邻wave跨expert反而把这些访问打散，且expert-minor映射还
  引入runtime quotient/remainder。候选不仅未省10us，最佳也倒退26.875us，因此按门槛
  停止，不再做gate版本、不接selector/service/production。

### C4短上下文sequential top-k zero-fill删除反例（2026-08-27）

- `seq_len <= index_topk=512`时，ROCm的SGL top-k transform进入
  `naive_paged_transform`，不读取score tensor。以`torch.empty`替换仅限
  `sgl-kernel` backend的`torch.zeros([B,512])`后，BS1--32、page-size
  1/4/16/64/256共30组physical/raw indices均逐元素exact。
- BS32 CUDA-graph component ABBA从`zero_ + sequential transform=6.460 us`
  降至`empty + transform=5.269 us`，局部省`1.191 us/C4 layer`。
- 但同代码/同启动参数（TP8/EP1、graph tiers 1/32）的两次独立服务均通过France
  tier1+32合计`33/33` exact后，128-token、32并发多轮结果为：baseline正常轮中位约
  `877.7 tok/s`，empty候选正常轮中位约`875.6 tok/s`，候选约`-0.24%`。两边各有
  服务级慢轮，局部memset节省明显低于端到端波动且没有净收益。
- 已恢复`torch.zeros`，不提交production改动。以后不能把“不读取logits”直接折算成
  服务收益；除非能与整个sequential transform/metadata写入一起消除，否则不重做。

### TP8并发hash漂移的prefill来源复核（2026-08-27）

- 同一16-token显式`input_ids`、temperature 0下，串行BS1的前4个输出32/32均为
  `[19,16,223,2619]`；32个同时提交的请求则常分成`[...455]`与`[...2619]`两条轨迹。
  首次token-ID分叉在generated position 3，而不是SWA=128边界。
- 但进一步抓`max_new_tokens=1`的首个输出top-5 logprobs时，32请求在prefill最后位置
  已经有16种logit pattern；top候选虽然都为token 19，后续候选margin可相差约0.5--1.0。
  因此长输出hash差异由并发prefill的不同batch/shape数值路径先产生，再被greedy自回归
  放大，不能当作decode graph row race或SWA cache bug证据。
- 单变量关闭custom all-reduce、改用RCCL后，串行BS1仍8/8稳定，而BS32仍按
  `22/10`分成两条前4-token轨迹，排除AIter peer-read AR为该现象的根因。两种collective
  的France tier1+32均`33/33` exact。保持性能更好的custom AR；后续decode优化必须继续
  使用固定token teacher-forced/layer oracle或大margin France oracle，不能要求不同
  prefill batching的长greedy hash天然一致。

### TP8 M32 projection-owner通信oracle否证（2026-08-27）

- 使用真实layer20 `attn_norm [32,4096]`和四个production逻辑BF16矩阵，新增独立
  projection-owner oracle。reference A在每个rank仍串行执行原shape
  `F.linear`：N=`1536/2048/512/64`；candidate只让rank0--3各执行其中一个完整
  原shape GEMM，rank4--7不执行projection。没有修改production selector或模型代码。
- 首版fixed `[M,2048]` registered all-gather保持输出bitwise exact，但candidate
  critical path约`187.6 us`，padding传输和rank-major repack已超过收益预算，未通过
  `>=30 us/layer`门槛。
- 随后新增peer-read版本：owner将完整原shapeGEMM输出复制到AIter direct
  `allocate_meta_buffer`注册槽，以system-scope release发布epoch；每个consumer等待
  四个owner并使用gfx90a GLC peer loads只读取`1536/2048/512/64`有效宽度，直接pack
  `[32,4160]`。每轮consumer ack后owner才允许覆盖，另有8-rank end epoch限制graph
  replay漂移。实现位于`gfx90a_projection_owner_peer_oracle.cuh/.py`及
  `scripts/rocm/bench_dsv4_tp8_projection_owner_ag.py`，均为独立oracle。
- correctness通过：1000种真实输入mutation在8 rank上`0/1000` mismatch、最大误差
  `0`；另做1000次固定输入graph replay同样`0/1000` mismatch。A/B在所有rank的SHA256
  均为`49abcf56943d07a241cfe0ab6e8663af1f04930c1d61cc5ee3f2ba9045b7c88c`。
- 7轮rank-max ABBA稳定判负：A1=`123.927 us`、B1=`135.451 us`、B2=`135.122 us`、
  A2=`123.907 us`；paired mean为A=`123.917 us`、B=`135.286 us`，candidate反而慢
  `11.370 us`（`+9.18%`）。owner-only约`40.656 us`，说明剩余publication、peer pack
  和全rank epoch固定成本大于省掉的三个projection。
- 结论：即使消除all-gather padding并保持原GEMM reduction tree，M32下按projection
  owner分工仍没有收益；它未达到30us门槛，不接service/production，也不再继续调
  peer pack几何。后续attention优化应保留各rank本地projection及现有异步分支。

### TP8 projection-owner v2多CTA复核（2026-08-27）

- 将peer pack从单CTA改成同stream的`wait(1 CTA) -> copy(65 CTA) -> ack(1 CTA)`，仍保留
  owner overwrite和8-rank graph epoch协议。1000种输入mutation与1000次graph replay
  在八rank均为bitwise exact，最大误差`0`且所有rank SHA256一致。
- 7轮rank-max ABBA从reference `123.984 us`降到candidate `76.424 us`，oracle净省
  `47.560 us/layer`（`-38.36%`），owner compute-only为`40.436 us`，因此通过了
  `>=30 us/layer`的production试接门槛。
- 默认关闭的临时production接线仅命中`gfx90a + TP8 + C4 + native decode M32`；只捕获
  graph tiers `1/32`。graph捕获稳定，France为BS1 `1/1`、BS32 `32/32`逐tokenexact且
  唯一。BS32x256候选前三轮为`956.018/954.419/959.472 tok/s`；同配置关闭候选的回程
  首轮为`944.586 tok/s`。候选最多只有约`1--1.5%`服务收益，远低于5%保留门槛，也未
  稳定突破1000；每个C4层还新增2MiB registered workspace。
- 结论：多CTA解决了独立peer pack固定成本，但完整graph中的跨卡publication、CU/cache
  竞争仍吞掉大部分oracle收益。production接线已撤掉；只保留独立oracle及其exact/性能
  证据，不把micro结果外推为模型吞吐。

### TP8-attention + MLP-DP2/TP4 compute下界（2026-08-27）

- 为避免重复完整attention/KV，单独评估只把MLP的32行分成两个并行16行组：现路径每GCD
  是`M32/I256`专家shard，候选每个四GCD组是`M16/I512`。新增独立脚本
  `scripts/rocm/bench_dsv4_tp8_mlp_dp2_lower_bound.py`，候选尚未计入每层必要的两组
  `16x4096 BF16`输出交换，因此是对hybrid结构有利的compute-only下界。
- 相同A4/R2/W8/LDS full routed stage，当前`M32/I256/B832`为`272.689 us`；候选扫描
  gate/down blocks `416/624/832/1040`后最佳`M16/I512/B416`仍为`277.185 us`，已慢
  `4.496 us`（`+1.65%`）。其他候选为`280.107--284.908 us`，慢`2.72--4.48%`。
- 原因是TP4把每rank expert intermediate shard从256翻倍到512；两组并行虽把token行数
  减半，但单GCD权重/算术工作并未下降，M16也不足以提高复用。加入跨组row exchange只会
  更慢且额外复制约17GiB routed权重/GCD，因此不改scheduler/weight loader。

### TP8 output-N复接、长轨迹与组合开关复核（2026-08-27）

- 为核对旧1000 tok/s高点，重新以默认关闭开关接入C4/C128 M32 output-N：全部BF16
  projection weight在quant postprocess完成后的graph warmup懒建N-shard cache，真正
  capture内首次分配会fail-loud；每rank执行N520(C4)/N320(C128) full-K GEMM，再用已有
  AIter custom AG恢复bundle。修复了最初在`post_load_weights`过早读取qkv FP8 raw
  weight的问题。
- graph tiers1/32稳定；France为BS1 `1/1`、BS32多轮`32/32`逐tokenexact。候选串行
  512-token SHA256为
  `d346ee6f8e3be250ec12cba660d0e9dfd0df2f1c0ef51a9d867dc2d6a0098ad7`；关闭候选的
  独立服务得到完全相同512个token与同一SHA256。虽然独立projection中间tensor仍有
  已知约1e-4 relative-L2差异，这次长greedy轨迹没有分叉。
- 当前机器六轮BS32x256为`984.952/980.515/993.140/985.240/987.286/986.379 tok/s`，
  中位约`985.8`；相对同日baseline约`944--962`有约2.5--4.4%收益，但未稳定破1000，
  也未到5%正式checkpoint门槛。恢复完整graph tiers后六轮仅
  `975.229/982.330/981.657/978.551/981.414/981.417`，不是缺失tier造成。
- `SGLANG_DSV4_GFX90A_REPLICATE_EMBEDDING=1`与output-N组合的512-token输出逐token相同，
  但六轮正常中心只有`981--984 tok/s`且一次落到`799.335`，没有收益。ROCm multistream
  关闭后France仍32/32 exact，但六轮仅`894.693--902.125 tok/s`，约退化9%；bundle AG
  后的q/core/index consumer仍必须保持多流。
- 负载中八GCD均观测到1700MHz，DVFS不是约2%差距来源；固定HIGH需要sudo，当前账户无
  passwordless权限。对M32/N320/K4096枚举2226个hipBLASLt solution，最快约`43.97 us`，
  而当前`F.linear`已约`33.07 us`，显式定解无收益。output-N生产分支暂保持默认关闭，
  只用于继续组合；当前证据不能报告为稳定破1k checkpoint。

### output-N BS32 graph-replay marker复核（2026-08-27）

- 在layer20启用默认关闭的`REALTIME_TRACE`，只捕获tiers1/32并每64 replay读回一次。
  logger的跨八rank device-to-host同步将本轮HTTP吞吐压到`582.9 tok/s`，该数字仅为诊断
  副作用，不能覆盖无marker的约986结果。
- 多组完整M32 replay的最慢rank单层span约`718--727 us`。粗段中attention入口MHC约
  `46--52 us`、attention prepare约`84--93 us`、后续attention/output约`69--74 us`，
  FFN入口MHC/归一化合计约`205--220 us`，MoE区约`311--327 us`。
- MoE细分明确为：router projection约`26--30 us`、top-k约`12--13 us`、routed experts
  约`218--237 us`、join/add小段各约`4 us`、最终TP8 all-reduce/尾部约`31--44 us`。
  因此下一项可叠加收益必须来自routed stage或router/topk/sort边界，而不是继续削已经
  约10us级的attention小尾核。output-N从约986破1000只需约12us/layer，但1500仍需
  结构性减少weight scan/collective固定成本。
- 独立graph确认M32 router的`tgemm`与`F.linear`逐元素exact，二者均约`12.29 us`；
  Python eager看到的`73 vs 36 us`只是dispatcher host overhead，不存在graph内替换收益。
  2226个hipBLASLt solution也没有快于当前captured kernel的候选。
### 单TP4、真实多样BS32的A4/LDS routed几何（2026-08-29）

- 目标口径改为单实例`TP4/EP1/no-A2A`、4 GCD、native AR、最多BS32，权重保持原始
  checkpoint格式。测试输入使用
  `.agents/memory/dsv4_tp8_diverse_32_input_ids.json`中的32条不同固定token-ID请求；
  每轮独立`cache_salt`，不是重复prompt。
- 旧`SGLANG_DSV4_GFX90A_MULTI_REQUEST_THROUGHPUT_PROFILE=1`实际沿用双TP4、每副本
  M16调出的`A8/rows2/624 blocks`，并且没有启用LDS E2M1 LUT。新增production-shape
  oracle`scripts/rocm/bench_dsv4_tp4_m32_grouped_oracle.py`，使用真实diverse recorder
  的layer34路由、TP4 `E256/H4096/I512/topk6`，完整计入gate、第二次group32 quant、
  down FP32 partial和固定slot reduction。
- production-exact no-LDS A8/624为`710.301 us`；开启LDS后A8/624为`564.532 us`。
  A4/rows2/832在两次独立运行中为约`451.3 us`，另一受邻卡实验扰动的三轮中为
  `449.731/561.452/562.764 us`；A4/rows4通常约`470.9 us`。所有profile最终BF16
  输出相对no-LDS基线逐元素bitwise exact、`max_abs=0`。
- 只捕获tier1/32时，真正命中BS32 graph的多样请求128-token轮为`610.825 tok/s`，
  相对旧完整tier服务稳态约`498.8--500.2`提升约22%；256-token命中轮为
  `629.263 tok/s`。稀疏tier会在请求入场/结束错位时落eager，产生130--323 tok/s
  慢轮，不能交付。
- 捕获1--32全部tier后，32条不同请求、每条256 token的完整BS32波次为
  `622.388/630.336/630.369 tok/s`，France首9 token精确、全部请求长度256。
  HTTP总wall仍会因page-size256把32条短prompt按8条分批prefill而出现2--3个decode
  波次；这属于admission/prefill口径，后续应以长驻留decode window或增大prefill chunk
  分离，不能把整数波次总wall误判为kernel退化。
- 新增显式`SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1`：选择TP4/EP1/no-A2A、A4、
  LDS LUT、rows2、gate/down 832 blocks，并捕获1--32全部tier。它不覆盖旧双TP4 M16
  profile，因为历史M16 A4端到端已判负。
- 后续TP4 K512 down-consumer quant融合扩展了既有TP8 oracle：同一512-thread CTA的
  32个16-lane subgroup各顺序量化两个group，保持group32除法/cast、down partial及
  fixed-slot reduction次序。CTA4/8/12/16的partial和最终BF16均逐元素exact；100次
  mutation和100次graph replay exact。最佳CTA16完整routed仅`449.428 -> 443.112 us`
  （`-1.41%`），远低于`<=410 us`门槛，不接production selector。
- A8/rows1虽减半accumulator，624/832/1040/1248 blocks最好仅约`504.8--522.9 us`，
  仍慢于A4/R2约12%以上；不重访A8。显式`uint4` packed-weight load与gate/up成对SDOT
  helper都让A4保持约`449--450 us`，证明hipcc已有等价load/CSE，源码均已撤销。
- 将gate/down grid解耦后，down扩大到1040/1248明确退化；gate从832扩大时，完整stage
  为g1040/d832 `444.385 us`、g1248 `442.529`、g1872 `441.529`、g2080
  `441.513 us`，全部bitwise exact。只捕获1/32的完整服务上，g2080热稳态为
  `729.878/732.760/731.938/732.667 tok/s`，相同重复请求g832约`722.7--726.4`，
  端到端约+1%；因此单TP4-BS32 profile保留gate2080/down832的小收益。
- 进一步把gate rows2固定、只切down rows1/2/4做干净GCD0复测：g2080/d832的
  rows2为`439.060 us`，down rows4为`439.452 us`，rows1为`455.252 us`；rows2已经
  最优，先前gate/down同时rows4没有掩盖down收益。
- 按CK `WarpGemmMfmaBf16Bf16F32M4N64K16`和CDNA2
  `V_MFMA_F32_4X4X4BF16_1K`建立过独立TP4 A4 gate原型：wave64的16个N-block×4 lane
  精确承载四assignment，同wave维护gate/up accumulator，直接读取原packed FP4并按
  E8M0 group32在线解为BF16；没有改权重或写常驻repack。tiles-per-wave 1/2/4、
  blocks104/208/416均finite，但最快tiles1仍约`11.0 ms`，对production gate约
  `256 us`慢约43倍；tiles2/4约12/22ms。K4096下每wave需要2048条gate/up M4 MFMA，
  再叠加逐K4 FP4→BF16转换，完全抵消无padding优势。候选同时相对INT8 production
  gate约4.35% relative-L2（尚未做raw-BF16 reference）；性能已远低于15%继续门槛，
  因此原型删除、不实现down、不接selector。此结果明确补齐旧M16/M32 padding反例之外
  的M4空白，后续不得再以“恰好A4无padding”为由重做BF16 M4 MFMA。

### TP4 A4 gate kRows=2 cross-row activation reuse oracle (rejected)

- An isolated, non-production kernel explicitly loaded each assignment's 32-byte
  `xq` group once and reused it across row0/row1 gate/up SDOT chains.
- All five block profiles were BF16 bitwise exact (`max_abs=0`).  The existing
  gate2080/down832 path measured 441.694 us, while reuse gate blocks
  832/1248/1664/2080/2496 measured 560.897/548.897/560.417/552.417/557.616 us.
- Best reuse was 24.3% slower, consistent with extra VGPR pressure reducing
  occupancy while the original code/compiler already reuses activation loads.
  Do not repeat this explicit cross-row register-reuse direction.

### TP4 BS32 rank-max marker and DPP reduction ABBA (2026-08-29)

- Current target is one TP4/EP1/no-A2A replica on four GCDs, native AR, original
  checkpoint weights, and 32 distinct fixed input-ID requests. Layer20 (C4)
  slowest-rank median was `1151.84 us`: attention-entry MHC `86.24`, prepare
  `230.08`, attention core `56.32`, output `131.84`, FFN-entry MHC `91.44`, and
  MoE `552.96 us`. Inside MoE, router/top-k/routed/TP4-AR were approximately
  `26.8/11.68/465.52/33.52 us`. Layer21 (C128) was `1063.76 us`, with prepare
  `160.48` and routed `463.60 us`; thus C4 compressor adds about 70 us on its
  layers, but routed FP4 remains the common critical section.
- An isolated CDNA2 DPP reduction variant kept gate offsets32/16 as shuffle-down
  and replaced offsets8/4/2/1 with row-shift DPP; K512 down used DPP for all four
  subgroup16 steps. One hundred captured replays were final-BF16 bitwise exact.
  Seven-round oracle ABBA improved the complete routed stage from trimmed
  `439.337` to `423.373 us` (-15.968 us, -3.64%).
- Production A/B used a streaming diverse-request harness. Its resident window
  begins after all 32 requests emitted a token and ends when the first request
  emits its last token, so prefill/admission is reported separately. With
  identical 512-token generation lengths, all France first-nine-token checks
  passed; resident A1/B/A2 were `595.996/412.414/596.861 tok/s`. The DPP code
  therefore caused a reproducible 30.9% graph/service regression despite its
  micro win, consistent with changed occupancy/stream scheduling. Production
  wiring was removed. Keep the isolated oracle but do not enable DPP in TP4.

### TP4 M32 token-row-owned MHC component oracle (2026-08-29)

- Tested the strongest exact-native proposal from the external review without
  changing production: four TP ranks either each execute the existing MHC
  backend over all 32 rows (A), or each execute the same backend over its owned
  contiguous 8 rows and publish normalized hidden rows with the real registered
  AIter all-gather at both attention and FFN boundaries (B). Incoming
  reduce-scatter was deliberately excluded, making B an optimistic bound.
- Eager local hidden/residual/post/comb slices and gathered M32 hidden were all
  bitwise exact. The captured four-rank path passed 1000 replay iterations with
  no stale output or hang. Seven-round rank-max ABBA gave A1/A2 centers around
  `106.631 us` for two boundaries and B1/B2 `133.979 us`; row ownership was
  `27.348 us` slower. Compute-only row ownership was `84.363 us`, saving only
  `22.267 us`, while the two publications added `49.616 us`.
- This fails the predeclared `55--60 us/layer` continuation gate even before
  accounting for an incoming reduce-scatter. Do not add production row-owned
  state or graph-tier protocol. The reusable oracle is
  `scripts/rocm/bench_dsv4_tp4_token_row_mhc_oracle.py`.

### TP4 grouped-FP4 gate/down DPP isolation (2026-08-29)

- Split the exact DPP candidate into A=`shuffle gate + shuffle down`,
  G=`DPP gate only`, D=`DPP down only`, and B=`DPP both`. One hundred mutated
  activation/router-weight cases matched intermediate BF16, FP32 partial, and
  final BF16 bitwise; G and D each passed 1000 captured replays at all three
  boundaries. Seven-round micro ABBA gave full routed A/G `439.353/426.569 us`
  (-12.784), A/D `439.283/432.689 us` (-6.594), and A/B
  `439.121/422.867 us` (-16.254).
- B had already regressed diverse resident service by 30.9%, so production A/B
  tested only G. With identical 32-distinct-input, 512-token streaming resident
  windows, A samples were `596.861/592.956/593.571/596.919 tok/s`; two
  independent G services were `601.712/598.117`. G1 versus adjacent A1/A2 was
  +1.14%, but G2 versus A3/A4 only +0.48%; aggregate means were about
  `595.08 -> 599.91 tok/s` (+0.81%). Every round passed the France first-nine
  token oracle and generated the full length.
- Gate-only DPP is a small real micro win and a weak service trend, but it fails
  the predeclared stable >=1% service gate. Production selectors were removed;
  keep the A/G/D/B oracle only. Do not combine gate+down DPP in the TP4 graph.

### TP4 M32 C4 attention multistream checkpoint (2026-08-29)

- The existing HIP multistream prepare path had remained unreachable and still
  addressed the legacy raw-SWA pool. A narrow default-off selector now permits
  only `gfx90a + attn-TP4 + C4 + native decode M32 + graph capture`; unified-KV
  uses `get_unified_kv`, unified cache locations, page size 1 and BF16 store.
  Other graph tiers and C128 layers retain the serial path.
- The first capture exposed the stale raw-SWA dependency (`swa_kv_pool=None`),
  which was fixed rather than bypassed. Seven valid four-rank coarse marker
  groups then reduced C4 prepare from serial median `230.08 us` to `148.16 us`
  (range `147.36--153.60`), saving `81.92 us/C4 layer`. The old fine marker
  validator initially saw a stale slot15 because that marker was absent in the
  multistream branch; slot15 is now written after the indexer consumer join.
- Streaming resident ABBA with 32 distinct fixed input-ID prompts and 512 output
  tokens gave serial A=`592.928 tok/s`; independent B services were
  `613.866/613.277 tok/s`. Restoring all graph tiers 1--32 produced
  `615.467 tok/s`; all 32 requests completed 512 tokens with `finish=length` and
  France first-nine exact. Relative to adjacent A this is about +3.5--3.8%.
- A single parallel-batch M32 teacher-forced oracle compared serial and both
  quick/full-tier candidates: 32/32 next-token IDs, top-5 IDs, token logprobs
  and complete returned rows were bitwise identical (`max logprob abs=0`). The
  reusable checker is `scripts/rocm/check_dsv4_tp4_m32_next_token.py`.
- Enable `SGLANG_DSV4_GFX90A_TP4_M32_ATTN_MULTISTREAM=1` by default only inside
  the explicit TP4 BS32 profile. This is an exact native-AR scheduling change;
  it does not alter weights, attention semantics, or speculative decoding.

### TP4 M32 C128 attention multistream rejection (2026-08-29)

- Reused the exact C4 three-stream schedule for C128 layers behind a separate
  default-off selector. Layer-21 four-rank markers reduced prepare only from
  the serial `160.48 us` baseline to roughly `140 us`, about `20 us/layer`.
- The no-marker service with the already accepted C4 overlap plus the C128
  candidate produced `586.316 tok/s` in the resident BS32 window, below the
  adjacent C4-only range of `592.956--596.919 tok/s`. All 32 requests still
  generated 512 tokens and the France first-nine-token check passed.
- The extra C128 stream/event joins therefore cost more than the hidden
  compressor work. The C128 selector and production wiring were removed; do
  not generalize the C4 multistream result across compression ratios.
- The diverse resident harness now accepts `--stream-interval` so host/HTTP
  notification cost can be diagnosed independently. Interval 8 measured
  `607.479 tok/s`, but formal comparisons remain at interval 1 because coarse
  notification shifts the observable common-resident boundary.

### TP4 M32 attention-output local oracle (2026-08-29)

- Isolated the C4 output chain (`wo_a`, layout restoration, `wo_b`) without
  production wiring. Splitting M32 rows across two/four/eight streams measured
  `94.10/141.45/271.81 us` versus the serial `73.99 us`, regressions of roughly
  27/91/267%. The changed M shapes also changed GEMM reduction order
  (`max_abs=0.0078125`), so this is neither fast nor bitwise exact.
- Replacing the grouped einsum with a strided batched `bmm` was bitwise exact,
  but the complete output chain improved only `73.99 -> 73.56 us` (+0.58%).
  This is far below the predeclared 12-us/layer continuation threshold.
- Keep `scripts/rocm/bench_dsv4_tp4_attn_output_row_pipeline.py` as the exact
  oracle. Do not add row-pipeline streams or a production bmm selector; the C4
  attention-output tail has no locally demonstrated >=1% service candidate.

### TP4 diverse-routing gate-grid revalidation after C4 overlap (2026-08-30)

- Rechecked whether the production `gate_blocks=2080` choice had been biased by
  its original repeated-prompt service test. Both services used the accepted C4
  multistream path, graph tiers `1/8/16/24/32`, 32 fixed distinct input-ID
  prompts, 512 native-AR output tokens and the common-resident BS32 window.
- `gate_blocks=832` measured `610.670/610.643/611.815 tok/s`; the adjacent
  `gate_blocks=2080` service measured `613.982/614.657/615.128 tok/s`.
  Every round completed all 32 requests and passed the France first-nine-token
  check. The larger gate grid remains about 0.6% faster even with diverse
  routing and the new C4 stream overlap.
- Keep 2080 as the TP4-BS32 profile default. Do not add a route-dependent grid
  selector: the measured difference is small, but its sign is stable and the
  lower-CTA alternative does not recover hidden overlap.

### TP4 M32 fused gate-quant plus A4-sort service rejection (2026-08-30)

- Enabled the existing `gfx90a_m32_quant_sort` single-launch candidate only for
  the exact TP4 M32/A4 shape. It replaces separate group-32 INT8 gate-input
  quantization and expert sorting, without changing weights or routed math.
- A parallel 32-distinct-input teacher-forced request matched the serial path in
  all 32 output IDs, complete returned logprob rows and top-5 entries bitwise.
- With C4 multistream and gate/down grids 2080/832, three 512-token diverse
  resident runs measured `612.454/612.582/611.227 tok/s`. The adjacent unfused
  service measured `613.982/614.657/615.128 tok/s`; fusion therefore regressed
  about 0.4--0.6% despite removing one launch.
- Keep `SGLANG_DSV4_GFX90A_M32_FUSED_QUANT_SORT=0`. Its combined CTA/LDS work
  does not improve the production graph, so do not enable it merely from launch
  count reasoning.
- Revalidated this conclusion after the exact DPP-gate/down-prefetch checkpoint
  and with scheduler-reported model throughput, eliminating the old HTTP-tail
  ambiguity. Three 512-token rounds with 32 distinct inputs passed France,
  length and the complete 32-row teacher-forced token/logprob/top-5 comparison.
  Fused quant-sort measured model-decode median `705.475 tok/s` (trimmed mean
  `705.889`, range `702.86--710.49`) and HTTP resident
  `619.621/619.054/619.344`. The adjacent selector-off checkpoint had two
  independent model-decode medians `707.990/706.140`; the fused path remains
  neutral-to-negative even under the corrected metric. Keep it disabled.

### TP4 grouped-FP4 E8M0 half-scale ISA oracle (2026-08-30)

- Tested direct construction of the E8M0 half-scale (and an edge fallback) so
  the hot grouped gate/down dot would nominally replace
  `x_scale * weight_scale * 0.5`. For real quant activation scales, exponents
  7--254 were bitwise identical; the guarded candidate then kept gate BF16
  intermediate, down FP32 partial and final BF16 output exact across 100 input
  mutations. Three artificial extreme float cases still differed outside the
  real quant-scale domain.
- Seven-round ABBA nevertheless regressed: gate trimmed
  `255.997 -> 259.725 us` (+1.46%), down `171.614 -> 172.227 us` (+0.36%), and
  full routed stage `441.754 -> 444.336 us` (+0.58%). Production was never
  wired and the oracle source changes were removed.
- HSACO explains the result: baseline and candidate both contain 24
  `v_mul_f32`, 21 `global_load_dwordx4`, and 64 `ds_read_b32`; the candidate
  adds four compares and 41 disassembly lines. hipcc already folds the original
  scale expression. It also CSEs gate/up activation loads: each group has only
  two dwordx4 activation loads (offsets 0/16) shared by both projections.
  Do not revisit half-scale construction or explicit gate/up xq-load sharing.

### TP4 M32 down-consumer service recheck after C4 overlap (2026-08-30)

- Re-enabled only the existing exact CTA16 down-consumer candidate under the
  accepted C4 multistream configuration. It quantizes BF16 expert intermediate
  inside the down CTA and retains the fixed-slot FP32 partial/reduction order.
- The 32-distinct-input teacher-forced response matched baseline output IDs,
  complete logprob rows and top-5 entries bitwise. Three 512-token resident
  runs measured `613.705/614.753/614.740 tok/s`, versus adjacent baseline
  `613.982/614.657/615.128 tok/s`.
- The centers differ by less than 0.1%; the old 1.41% TP4 component saving is
  fully hidden in the current service graph. Keep
  `SGLANG_DSV4_GFX90A_M32_DOWN_CONSUMER=0` rather than adding a neutral path.
- Rechecked it once more on top of the accepted DPP-gate/down-prefetch
  checkpoint using scheduler model throughput. Down-consumer replaces the
  row-prefetch down path while retaining the DPP gate. The 32-row
  teacher-forced token, full logprob and top-5 rows were bitwise exact; three
  512-token rounds passed France and length. Model-decode median was
  `706.640 tok/s` (trimmed mean `706.553`, range `700.08--711.20`) and HTTP
  resident was `618.827/618.549/618.364`, within the two independent current
  baseline centers `707.990/706.140`. It remains neutral and stays disabled.

### TP4 high-priority auxiliary-stream rejection (2026-08-30)

- Gave HIP auxiliary stream 0 high priority behind a temporary default-off
  selector. This stream carries the C4 core compressor and SBO shared expert;
  the hypothesis was that shorter auxiliary join tails might outweigh added
  contention with main-stream attention and routed experts.
- The 32-distinct-input teacher-forced response remained bitwise identical.
  Three 512-token resident runs were `614.953/614.292/615.990 tok/s`, versus
  adjacent normal-priority baseline `613.982/614.657/615.128 tok/s`; median
  movement was only about +0.05%.
- HIP stream priority did not change the production critical path materially.
  The selector and construction change were removed; keep normal priority.

### TP4 M32 all-reduce plus MHC-post structural oracle (2026-08-30)

- Built an isolated four-rank oracle for the proposed AR-to-MHC boundary. The
  debug reduction exactly reproduced production's TP4 two-stage BF16 result,
  including its flat-buffer owner rotation: rows 0--7 sum ranks `0123`, rows
  8--15 `1230`, rows 16--23 `2301`, and rows 24--31 `3012`. This is why a
  conventional H-dimension reduce-scatter is not bitwise equivalent.
- Production component timing was already a hard upper-bound rejection:
  AR-only was `26.18--26.40 us`, MHC-post-only `9.63--9.76 us`, and their
  captured chain `32.33--32.63 us`. Even deleting MHC-post for free cannot meet
  the declared 20-us/boundary continuation threshold.
- A direct 64-CTA one-stage peer-read fusion measured
  `157.57--158.52 us`; repeated entry/exit cross-rank barriers dominate. Its AR
  debug output was bitwise exact under mutations, but the HIP MHC association
  differed from production Triton by up to `6.1e-5` in BF16 output and about
  `1e-6` in RMS partials.
- A stage1-only RS plus token-CTA peer-pull was therefore not extended: it must
  add remote reads, publication and a final reuse barrier while its absolute
  upside is below 10 us. Keep the independent oracle files for reproducing the
  exact owner-order and latency bound; do not wire this protocol into the model.

### TP4 M32 native full-MHC gate audit (2026-08-30)

- The existing wave64 HIP `gfx90a_mhc_post_pre` accepts arbitrary token counts,
  although production only selects its full path for global BS1. A new oracle
  compared it directly against the production M32 Triton decomposition using
  the real rank0/layer20 FFN boundary dump and 20 Sinkhorn iterations.
- Seven-round captured ABBA measured production `39.035 us` and native HIP
  `45.444 us`; the one-CTA-per-token full kernel is 16.42% slower at M32.
- The path also changes arithmetic association: residual/post/comb/layer-input
  max-abs differences were respectively `0.001953125`, `1.98e-5`, `3.98e-5`,
  and `0.0078125`; final layer-input relative-L2 was `2.25e-4`.
- Therefore the `global_batch_size == 1` production gate is intentional, not an
  accidentally unreachable M32 optimization. Keep the oracle script, but do
  not broaden `SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE_FULL` to BS32.

### TP4 M32 MHC-RMS routed-gate prequant service recheck (2026-08-30)

- Temporarily wired the already exact MHC final-RMSNorm plus group-32 INT8
  producer into only the routed AIter gate path; shared experts continued to
  consume the original BF16 hidden state. Fused quant-sort and down-consumer
  remained off, so this isolated the first activation quantization.
- The 32-distinct-input teacher-forced response matched baseline output IDs,
  complete logprob rows and top-5 entries bitwise. Three 512-token diverse
  resident runs measured `614.985/615.188/614.962 tok/s`, versus adjacent
  baseline `613.982/614.657/615.128 tok/s`.
- The median movement is only about +0.05%. Although the component oracle saves
  about 36.7 us for RMSNorm plus quant, the separate quant is hidden by the
  current graph schedule. Production handoff/selector changes were removed;
  retain the existing standalone producer oracle only.

### TP4 M32 grouped-FP4 hardware-counter checkpoint (2026-08-30)

- Profiled the production-shape diverse route (`pass=37`, `layer=34`, 106
  active experts, 113 A4 scans) on one gfx90a GCD. Added `--profile` and
  `--waves` to the grouped oracle so rocprofiler can run one geometry without
  JIT-compiling and retaining every candidate.
- The uninstrumented A4/R2/W8/G2080/D832 full stage was `438.4--440.5 us`.
  Kernel trace measured gate/up about `259.36 us` (`96 VGPR`, `64 SGPR`,
  `1 KiB LDS`) and down about `170.24 us` (`52 VGPR`, `48 SGPR`, `1 KiB LDS`);
  group quant and fixed-order reduction remained about `5.63/5.20 us`.
- Performance counters show that essentially every external read request goes
  to DRAM. Gate/up issued about `3.72M` 64-byte DRAM requests (`~238 MB`) and
  down `1.87M` (`~120 MB`), corresponding to only about `0.92/0.70 TB/s` at
  their normal durations. L2 request hit rates were about `51.1%/35.7%`, while
  average external read residency was roughly `295/278` cycles. DRAM-credit
  stalls were below `0.1%` of summed TCC busy cycles. Thus raw HBM credits are
  not saturated; latency hiding, register residency, and packed-weight decode
  constrain effective bandwidth.
- The conda wrapper `/home/pc/anaconda3/bin/rocprofv3` mixes its packaged
  aqlprofile/HSA libraries with the system ROCm runtime and aborts counter
  collection (`aqlprofile API table load failed`). Use
  `/opt/rocm/core-7.14/bin/rocprofv3 --rocm-root /opt/rocm/core-7.14` with the
  matching `/opt/rocm/core-7.14/lib` first in `LD_LIBRARY_PATH`. TCC groups
  must be split into at most compatible small groups; an oversized group
  aborts with error 38.
- Reducing only gate rows from 2 to 1 lowered its code object from 96 to 80
  VGPR and kept all outputs exact. With W8/G2080 it reduced the full stage to
  about `435.4 us`, only `0.6--0.8%`. Increasing its grid cap to 3120/4160
  lost the gain. Combining rows1 with W4/G1664--2080/D1664 measured
  `438.2--445.6 us` versus adjacent W8 baselines `436.5--439.2 us`, so it was
  rejected. The small rows1 result is useful evidence but not a production
  checkpoint; the next representation must reduce real DRAM scans rather than
  only reshuffle occupancy.

### TP4 M32 row-tile-major grouped-FP4 task order rejection (2026-08-30)

- Tested a cache-only task permutation on the real diverse pass37/layer34
  route: map `task -> (row tile, expert block)` instead of production's
  `(expert block, row tile)`. It kept A4 accumulators, logical loads, partial
  layout and fixed reduction unchanged, while placing adjacent A4 blocks for
  one expert on nearby waves. Output was elementwise exact.
- Five rounds measured production expert-major at median `437.423 us` and
  row-tile-major at `677.518 us` (`+54.9%`). The permutation destroys the
  current contiguous row traversal and grid-stride balance; neighboring waves
  do not turn the small amount of same-expert traffic into useful cache reuse.
- The diverse 128-pass distribution averages about 106.72 active experts and
  113.37 A4 blocks, so only about 5.86% of blocks are second-or-later chunks of
  an expert. This strict TP4 result agrees with the older, more favorable TP8
  paired-A4 rejection. The experimental kernel/template code was removed; do
  not revisit row-major ordering without a fundamentally different load-sharing
  primitive and a substantially higher measured expert-run fraction.

### TP4 M32 pairwise-interleaved w13 layout rejection (2026-08-30)

- Repacked only the packed FP4 w13 rows from runtime `[all gate][all up]` to
  `[gate0,up0,gate1,up1,...]`, leaving the existing A16W4 scale shuffle,
  sorter, A4/R2/W8/G2080 geometry, w2 and fixed reduction unchanged. This
  reduces the physical gate/up row distance from about 1 MiB to 2 KiB without
  changing bytes read or arithmetic. The full output was bitwise exact.
- Seven rounds measured baseline median `436.817 us` and interleaved median
  `436.305 us`, only `0.512 us` (`0.12%`) faster. The locality change does not
  materially improve the DRAM-bound gate/up critical path and misses the
  10--15 us continuation gate by a wide margin.
- Production would also need to replace, rather than duplicate, roughly
  22 GiB/GCD of w13 storage across 43 layers and carefully preserve the
  independently interleaved scale layout. All experimental code was removed;
  do not add a runtime weight-layout tag or loader repack for this result.

### Four-GCD PP4xTP1 routed lower bound and FP4 decode bounds (2026-08-30)

- Extended the grouped oracle with a local-intermediate-width selector and
  tested the only untried four-GCD structural layout that preserves per-GCD
  weight bytes: PP4 x TP1 with a static multi-microbatch decode conveyor. On
  the real diverse pass37/layer34 route, the TP1 `I=2048` routed stage alone
  measured `1667.0--1670.5 us`, median `1668.4 us` per layer. About 11 layers
  on the slowest PP stage already consume roughly `18.35 ms`; the entire
  BS32/1500 target step budget is `21.33 ms`, before attention, MHC, PP
  transport and bubbles. Do not implement PP4 scheduling until the packed-FP4
  stage is substantially faster.
- Removed the 1-KiB LDS FP4 pair-codebook while keeping the same inline
  `v_perm` codebook and A4/R2/W8/G2080/D832 work. The full stage remained
  bitwise exact but regressed from adjacent `435.5--436.8 us` to
  `636.8--637.0 us` (about +46%). The LDS LUT is essential; its shared-memory
  reads are much cheaper than reconstructing both sign paths in VALU.
- As a decode-free upper-bound diagnostic, expanded packed FP4 weights into
  exact signed INT8 codebook values while preserving E8M0 scale application,
  SDOT accumulation, sort, partial and reduction. Output was bitwise exact.
  Seven rounds measured packed baseline median `436.552 us` and pre-expanded
  INT8 median `627.582 us` (+43.8%). Doubling weight bytes costs far more than
  removing FP4 decode. Full INT8 prepack is both slower and incompatible with
  the memory target; retain packed 4-bit storage and the LDS-LUT decode path.

### TP4 M32 A1 plus A4-rest two-bucket rejection (2026-08-30)

- Closed the narrower gap left by the earlier TP8 A1/A2/A4 experiment. For
  real diverse pass37/layer34, split 64 singleton experts into an A1 grouped
  launch and kept all 42 non-singleton experts in 49 ordinary A4 blocks. This
  is an optimistic CPU-prepartitioned oracle; it excludes production GPU
  histogram/partition cost. Gate BF16, down FP32 partial and final BF16 were
  bitwise exact, including 100 mutated-input replays.
- A1/rest gate blocks `416/1664` and down `416/832` measured gate
  `255.534 -> 275.243 us`, down `172.869 -> 177.637 us`, and full routed
  `438.614 -> 463.960 us` (+5.78%). A higher-concurrency closure using gate
  `1040/2080` and down `832/832` improved the candidate but still measured
  gate `254.764 -> 268.886 us`, down `172.229 -> 176.037 us`, and full
  `439.763 -> 458.518 us` (+4.26%).
- The extra launch, per-grid LDS LUT initialization and synchronization exceed
  the benefit of smaller A1 accumulator state even though singleton blocks are
  56.6% of scans. It misses the predeclared gate continuation threshold
  (`<=239 us`) by about 30 us before sorter cost. Do not implement a two-bucket
  GPU sorter or service selector; occupancy partitioning without fewer weight
  scans is closed for this TP4 diverse workload.

### TP4 M32 packed-weight cache-policy rejection (2026-08-30)

- Tested cache policy only on packed FP4 weight loads; activation, E8M0 scale,
  metadata and LDS decode remained unchanged. Clang's
  `__builtin_nontemporal_load` was statically verified to emit coalesced
  `global_load_dwordx4 glc slc` on gfx90a, not an ignored hint. The real
  diverse full stage stayed bitwise exact but moved from median `439.082 us`
  to `443.130 us` (+0.92%). Bypassing/invalidation of useful L1 traffic is not
  beneficial.
- Also built identical 16-byte raw-buffer loads with explicit cache policy 0
  and SLC-only policy 2. Each variant ran independently and produced exact
  output. Raw policy 0 measured about `447.18 us`; raw SLC measured about
  `450.41 us`, versus adjacent flat/default around `439 us`. Thus changing
  addressing alone already costs roughly 1.8%, and SLC adds another 0.7%.
- gfx90a uses GLC/SLC semantics; gfx942 SC0/SC1 interpretations must not be
  copied here. The temporary CK descriptor and policy modes were removed.
  Default flat `global_load_dwordx4` remains the correct production choice;
  do not revisit cache flags unless a future compiler exposes an SLC-only flat
  load with identical addressing and code shape.

### TP4 M32 rows2 four-wave geometry closure (2026-08-30)

- Rechecked four-wave workgroups without the rows1 confounder. All candidates
  retained A4/R2 arithmetic and were bitwise exact. W4/G1664/D1664 measured
  about `439.46--440.50 us` versus surrounding W8/G2080/D832
  `437.20--438.82 us`, so it is neutral-to-slower.
- W4/G2080/D1664 was the only small win: two candidate centers were
  `432.529/432.881 us` versus adjacent W8 centers `436.855/436.876 us`, about
  `4.1 us` or `0.99%`. This misses the predeclared 3% component gate and is
  below the noise/graph-hiding budget established by prior service A/Bs.
- Keep these geometries in the standalone oracle for reproducibility, but do
  not add a runner waves selector or start a service experiment for this
  sub-1% component result.

### TP4 M32 compile-time constant LUT reconfirmation (2026-08-30)

- Reconfirmed the older constant-LUT rejection on the final TP4 pass37/layer34
  shape. Replaced each CTA's 1-KiB LDS pair table and barrier with an identical
  compile-time `__constant__` 256-entry table; packed weights, SDOT arithmetic,
  sort and reduction were unchanged and the result remained bitwise exact.
- Seven rounds measured LDS baseline median `437.207 us` and constant-LUT
  candidate `1117.737 us` (+155.7%). Divergent per-lane indices serialize or
  incur long scoreboard dependencies through the constant/global path; the
  repeated CTA initialization is far cheaper than every subsequent lookup.
- This agrees with the older M1 and grouped-M32 constant-table failures. The
  temporary symbol and kernel mode were removed. The per-CTA LDS pair LUT is a
  proven architectural requirement for the current packed-SDOT representation.

### TP4 M32 sequential gate-then-up residency rejection (2026-08-30)

- Tested a same-kernel two-pass gate/up mapping on the final A4/R2/W8/G2080
  shape. Gate completed its original K-order reduction first and lane0 stored
  the eight FP32 results in 256 bytes of wave-private LDS; up then repeated the
  same K-order and combined with the FP32 gate value before the single original
  BF16 output store. Packed weights, scales, SDOT and shuffle trees were
  unchanged. Intermediate and final routed outputs were bitwise exact.
- Seven rounds measured production full stage median `437.180 us` and the
  sequential candidate `535.162 us` (+22.4%). Kernel trace showed candidate
  gate about `362 us`, `88 VGPR`, `64 SGPR`, `1280 B LDS`, zero scratch. It
  removed only eight VGPR from the 96-VGPR baseline and did not cross the
  required <=64-VGPR residency tier.
- Re-reading activation/scales and repeating loop/address work therefore adds
  cost without enabling a second resident CTA. The candidate missed both
  static and timing gates and was removed. Do not revisit separated gate/up
  unless a representation demonstrably reaches <=64 VGPR without duplicating
  packed-weight or activation traffic.

### TP4 M32 FFN-boundary rank-max v2 and staged-MHC rejection (2026-08-30)

- Added the read-only diagnostic
  `scripts/rocm/bench_dsv4_tp4_ffn_boundary_rankmax_v2.py`. It parses seven
  accepted hot four-rank groups from the real-diverse-request layer-20 marker
  log and graph-replays the matching real M32 layer-20 tensor/weight dump. No
  production selector or model source was changed. The service rank-max
  medians were FFN-entry MHC `92.96 us`, router `28.64 us`, top-k `12.00 us`,
  routed expert `469.44 us`, and TP4 output all-reduce `36.16 us`. Four-rank
  AR-duration spread gives a conservative arrival-wait upper bound of only
  `5.28 us`; arrival skew is therefore not the main collective cost.
- The real-tensor explicit MHC decomposition measured post plus RMS partials
  `9.631 us`, pre-mix `23.946 us`, Sinkhorn `8.249 us`, weighted sum
  `8.282 us`, RMSNorm `8.340 us`, and the complete captured chain
  `39.747 us`. Component boundaries chained bitwise-exactly on all four ranks.
  These figures initially made the `92.96 us` service marker look like a
  roughly `53 us` production-backend opportunity, so the actual gfx90a
  `mhc_fused_post_pre` entry point was captured beside the explicit chain on
  the same stream and tensors.
- With the non-native Sinkhorn profile, seven-round four-rank ABBA gave
  production/staged medians `39.652/39.720 us`: staged was `0.068 us` slower.
  With the launcher's exact native-Sinkhorn and iteration flags, a confirming
  run gave approximately `41.757/41.860 us`, again making staged about
  `0.103 us` slower. Thus the direct production backend is already the same
  roughly 40-us graph sequence; no 20-us backend replacement exists. The
  `92.96 us` old service interval includes preceding stream/graph arrival
  dependencies or stale trace/profile context. Empty realtime-marker spans are
  about `1.4 us` each and cannot by themselves explain the difference, but it
  is invalid to attribute the full interval to MHC kernels.
- Across the original activation and 100 bounded teacher-forced hidden-state
  mutations, production and staged residual plus current `layer_input` were
  bitwise exact. Deferred `post` and `comb` were not exact: initial max-abs was
  approximately `0.00692/0.01480`, and mutation maxima were
  `0.00730/0.01625`. These states feed the next layer, so current-layer hidden
  equality is insufficient correctness evidence. The candidate fails both the
  `>=20 us` performance gate and deferred-state exactness; do not wire the
  explicit staged chain into production. Continue to treat routed FP4 MoE as
  the dominant exact-native TP4/BS32 target.

### TP4 M32 grouped-down R2 row-prefetch oracle (2026-08-30)

- Added an oracle-only A4/R2/W8/D832 grouped-down variant which requests both
  same-group R2 packed 16-byte weight rows and E8M0 bytes before decoding or
  consuming row0. It does not predecode weights, prefetch activations, change
  cache policy, or alter SDOT/shuffle/FP32 accumulation order. Baseline ISA had
  issued row1 only after completing row0 SDOT; candidate ISA emitted row0 and
  row1 `global_load_dwordx4` plus both scale requests before row0 LDS decode,
  retaining row1 in flight at `vmcnt(2)`.
- Static resources were unchanged at `50 VGPR`, `36 SGPR`, `0` VGPR/SGPR
  spills, `0` scratch and the same 1-KiB LDS LUT. On the real diverse
  pass37/layer34 route, 100 mutated activations/router weights preserved the
  intermediate BF16, down FP32 partial and final BF16 tensors bitwise exactly.
- Seven-round ABBA moved down trimmed mean from `172.325` to `170.342 us`
  (`-1.15%`) and full routed stage from `440.387` to `436.333 us` (`-0.92%`).
  The prefetch is real and correct but misses the predeclared continuation
  gates of down `<=127 us` and full `<=395 us`; do not add a production
  selector. Reusable files are
  `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_expert_down_row_prefetch_oracle.cuh`
  and `scripts/rocm/bench_dsv4_tp4_down_row_prefetch_oracle.py`.

### TP4 M32 exact-two-round gate-grid rejection (2026-08-30)

- Real diverse pass37/layer34 has 113 A4 scans and 256 gate row tiles, hence
  28,928 wave tasks. A 1,808-block/eight-wave grid executes exactly two tasks
  per wave, while the production 2,080-block grid leaves only part of the
  second grid-stride round active. This previously untested point checked
  whether tail imbalance, rather than memory latency, limited grouped gate.
- Seven-round identical-output timing gave full-stage medians of `446.368`,
  `444.635`, `440.843`, `441.568`, and `439.744 us` for gate block counts
  1792/1808/1824/1872/2080 respectively; every candidate was bitwise exact.
- Exact task divisibility is slower. The larger 2,080-block grid supplies more
  independent CTAs to hide the roughly 295-cycle external-read residency, and
  that benefit outweighs its partially occupied final round. Keep 2,080 as the
  diverse TP4 gate geometry; do not derive static grids from one route's task
  divisibility.

### Routed-FP4 lossless entropy and four-GCD PP2xTP2 bounds (2026-08-30)

- Sampled 144 original routed-expert weight/scale tensors across layers
  0/20/42 and eight experts per layer directly from safetensors. Packed E2M1
  codes have entropy `3.86035 bit/code` and only `6.386%` zero codes. The eight
  most frequent codes cover just `66.27%`, so a fixed 3-bit common-code plus
  escape representation would carry too many exceptions. E8M0 scales have
  entropy `0.9645 bit/scale`, but there is only one scale per 32 codes. Even an
  ideal zero-overhead entropy coder reduces combined weight-plus-scale bits by
  only `8.459%`; real random-access metadata and GPU variable-length decode
  would lower that gain. Do not build a lossless routed-weight entropy decoder
  for the current 1.5k target.
- A repository/history audit found no four-GCD `PP2 x TP2` DSV4 run. It is not
  flag-only: TP2/EP1 uses raw `w13=(256,2048,2048)` and
  `w2=(256,4096,512)`, absent from both the direct-AIter shape guards and the
  W2 inverse-layout guard. More importantly, halving resident layers while
  doubling each TP shard conserves per-GCD model bytes and KV bytes. The
  I1024 routed work is approximately twice TP4 per layer; the synchronous PP
  scheduler also disables overlap and historically made eight-GCD PP2xTP4
  only about `748 tok/s`. A four-GCD PP2xTP2 estimate is `337--400 tok/s`, with
  no capacity advantage over TP4. Do not add correctness-sensitive shape
  wiring for this throughput objective.

### Internal HIP custom-AR start-sync system-scope validation (2026-08-30)

- The internal AOT custom all-reduce start barrier now publishes each rank's
  readiness with a system-scope release store and polls its local peer flags
  with system-scope acquire loads. This establishes the missing cross-GCD
  happens-before edge from producer writes to the subsequent peer-buffer
  loads; the old relaxed store plus device-scope relaxed load did not express
  that ordering in the HIP memory model. The post-poll CTA barrier remains
  necessary to release the non-polling threads.
- Rebuilt for gfx90a with
  `cd python/sglang/kernels/aot && AMDGPU_TARGET=gfx90a MAX_JOBS=8
  /home/pc/anaconda3/envs/DS/bin/python setup_rocm.py build_ext --inplace`.
  The installed test artifact was
  `python/sglang/kernels/aot/python/sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so`
  with SHA256
  `d6bdd0e78c35e32638af26cddd659a19c59ede280ae16256a45d4bd633fbdb77`;
  the custom-AR object SHA256 was
  `c0ebb1328280f8975ed1c4c44cb5b14571f59cf531c61ddda90f4eac8f5e0fd0`.
  Note that the setup script re-hipifies generated `.hip` mirrors before its
  incremental Ninja build, so source-side `.cu` changes may appear as tracked
  generated-file updates and must be reviewed separately.
- `scripts/rocm/bench_internal_custom_ar_start_sync.py` captured the registered
  TP4 one-stage path for a real decode shape `[32,4096]` BF16 (256 KiB). Across
  1000 alternating random-normal/integer mutations and 1000 graph replays,
  every rank was bitwise exact against a CPU fixed `0,1,2,3` FP32 accumulation
  followed by BF16 conversion: failures `[0,0,0,0]`, maximum absolute error
  `0`. Nine rank-max samples were `29.721, 29.668, 29.678, 29.575, 29.528,
  29.557, 29.622, 29.586, 29.521 us`; the one-sample-per-tail trimmed median
  was `29.586 us`. This validates the isolated publication fix under graph
  mutation/replay; production service correctness and throughput still need a
  separate ABBA run before calling it a performance checkpoint.

### DSV4 C4-indexer long-context scaling oracle (2026-08-30)

- Added the service-free graph oracle
  `scripts/rocm/bench_dsv4_c4_indexer_long_context.py` for the model's actual
  indexer shape: batch 32, 64 index heads, head dimension 128, C4 page size 64
  and Top-512. Synthetic KV pages preserve the production structure-of-arrays
  byte layout (8192 FP8 value bytes followed by 64 FP32 scales per page), which
  is required even though the public tensor view is `[page,64,1,132]`. The
  current full-Triton kernel used its default `BLOCK_S=16`.
- The first seven-round alternating graph run measured the diagnostic
  `(full-Triton logits / torch Top-512 / logical-to-physical slot / complete
  torch chain)`: L513 `21.952 / 33.488 / 18.008 / 70.239`, L1024
  `34.680 / 47.671 / 17.784 / 98.327`, L4096
  `119.367 / 64.015 / 18.128 / 208.334`, and L16384
  `453.763 / 54.567 / 18.176 / 534.530 us`. This was only a diagnostic chain:
  production does not launch a separate torch Top-K followed by slot
  conversion. The AOT `topk_transform_512` v1 kernel already fuses Top-512,
  page-table lookup and physical-slot generation.
- A corrected production-path run measured `(logits / fused v1 tail / fused
  v1 full chain)` as L513 `22.400 / 14.208 / 31.440`, L4096
  `119.807 / 19.488 / 136.942`, and L16384
  `454.763 / 31.616 / 486.987 us`. The v1 slot sets were exact against the
  torch/page-table reference at every tested length. The optional v2 JIT path
  was unavailable on this gfx90a environment because its generated HIP source
  includes missing CUDA `cooperative_groups.h`; this does not affect the AOT
  v1 production path.
- Full-Triton scores agreed with the BF16 torch reference at maximum relative
  error `0.00150--0.00196` and mean absolute error about `0.066--0.068`.
  Top-512 set exact rows for L513/1024/4096/16384 were `32/27/20/8` out of 32;
  minimum set overlaps were `512/511/510/510` and mean overlaps
  `512/511.844/511.562/511.125`. Thus disagreements are only one or two entries
  at the Top-512 boundary under the expected BF16/Triton reduction difference,
  not gross score or addressing errors.
- The materialized FP32 logits write is 64.1 KiB, 128 KiB, 512 KiB and 2 MiB
  for the four lengths. A two-level design in which each CTA scans a 4096-token
  chunk and emits at most 512 `(FP32 score,int32 index)` candidates would write
  128 KiB through L4096 and 512 KiB at L16384. It is not a byte win at L513,
  ties at L1024, and reduces candidate/intermediate writes 4x at L4096/16384.
  However, the corrected production tail is only `19.49/31.62 us` at
  L4096/L16384, not the earlier diagnostic `torch.topk + slot` sum. The scan
  itself dominates at `119.81/454.76 us`; a candidate merge would consume much
  of the existing fused-tail budget. Do not claim a robust `>=20 us/layer`
  opportunity without a fused scan/local-select micro-oracle proving it. This
  is a long-context research path, not the explanation for the short 128-token
  versus 512-token service gap: C4 length is approximately full sequence/4,
  so Top-512 begins near an original sequence length of 2048 tokens.
- Production follow-up used the exact 32-request diverse manifest, forced
  native AR, graph tiers 1/32 and three 64-token rounds. The default external
  AIter two-stage service kept France correct and had resident BS32 samples
  `697.705/697.355/697.968 tok/s`, but only `10/32` requests were completion-ID
  exact across rounds. First divergence ranged from token 0 to 54. RCCL via
  `--disable-custom-all-reduce` was both slower (`440.860/441.281/440.254
  tok/s`) and still only `3/32` exact, so RCCL is not a correctness fallback.
- Before the publication fix, forcing `SGLANG_USE_1STAGE_ALLREDUCE=1` failed
  the first France gate with repeated stale token IDs. After rebuilding the
  release/acquire fix, France was correct and resident samples were
  `699.080/699.377/697.822 tok/s`; the one-stage path therefore restores its
  intended performance and no longer reads stale buffers. It still produced
  only `7/32` cross-round exact completions. The remaining variability is not
  the one-stage peer-buffer race or two-stage owner rotation alone: the three
  independent HTTP rounds reached prefill in different batch/chunk groupings,
  changing floating reduction/GEMM association before the fixed M32 decode
  graph. Fixed-batch teacher-forced comparison remains the kernel correctness
  oracle, while the concurrent harness now reports cross-round exact counts
  and first-divergence positions explicitly instead of hiding them behind the
  France-only check.
- Torch `/start_profile` with CPU+GPU activities is unsafe for this ROCm
  multi-stream graph: profiling began on EXTEND and HSA queue interposition
  waited indefinitely on async signals, stalling the requests. The affected
  service was stopped and all GPU resources released. Use external rocprof or
  the realtime-marker path for future BS32 traces.

### TP4 BS32 long-generation client-tail diagnosis (2026-08-30)

- Corrected an earlier shorthand: the roughly `699 tok/s` runs generated 64,
  not 128, tokens. A same-process A64/B512/B512/A64 test used the same 32
  distinct input IDs, graph tiers 1/32, fixed one-stage custom AR, native AR
  and stream interval 1. A64 resident medians were `696.735/697.378 tok/s`;
  B512 medians were `611.057/611.290 tok/s`. France remained exact and every
  request returned its requested length.
- Extending the harness with equal wall-time bins over the common BS32 window
  showed that this is not normal KV-length scaling. Across three more B512
  rounds, bins 0--5 sustained approximately `691--701 tok/s`, bin 6 delivered
  about `566 tok/s`, and bin 7 only `134 tok/s`; full-window samples were
  `611.147/611.757/611.706 tok/s`. Scheduler logs at 64-step cadence kept
  `#running-req: 32`, `cuda graph: True`, and reported model generation
  throughput `698.04--704.62 tok/s` through the decode. The client-side tail
  therefore measures output/detokenizer drain after GPU progress, not a model
  kernel slowdown. The stable native-AR model-decode checkpoint for this
  short-context TP4 profile is about 700 tok/s; retain HTTP aggregate and
  model-decode numbers as separate metrics.
- Global `--incremental-streaming-output` did not fix the tail: three B512
  resident samples were `608.927/612.430/610.630 tok/s`, France passed, and the
  same last-two-bin collapse remained. It also made two group-wall samples
  unusually poor, so it is not a production candidate.
- Four detokenizer workers initially failed correctness/availability before a
  valid round completed. `MultiDetokenizerRouter` asserted that a batch had
  invalid `http_worker_ipcs`: the ordinary Python HTTP frontend legitimately
  leaves each IPC name unset, while the multi-worker router incorrectly
  required it. The router now uses `http_worker_ipc` when present and otherwise
  hashes the request ID for stable detokenizer affinity, preserving the unset
  return route to the shared tokenizer manager. After the fix, 32x64 passed
  France and length checks at `696.882 tok/s`; three 32x512 rounds all passed
  France and length checks with aggregate `589.242/589.643/588.984` and
  resident `612.944/611.743/611.129 tok/s`. Thus the routing bug is fixed but
  four workers do not improve this client tail. Keep `DETOKENIZER_WORKER_NUM`
  optional, not a TP4 performance default. The benchmark retains optional
  token-position and common-wall-time bin reporting so future client-tail work
  cannot be mistaken for GPU decode optimization.

### TP4 M32 C4 output-N projection+AG oracle rejection (2026-08-30)

- Added standalone `scripts/rocm/bench_dsv4_tp4_output_n_projection_ag.py`
  using the real layer-20 M32 activation and four projection weights. Baseline
  A preserves the production GEMM boundaries N1536/N2048/N512/N64. Candidate B
  concatenates global N4160 weights, gives each TP4 rank one contiguous N1040
  full-K BF16 GEMM, runs AIter registered all-gather, then captures the
  rank-major `[rank,token,1040]` to consumer-major `[token,4160]` restoration.
  The qkv, core-compressor, index-compressor and index-weight segments were
  checked independently, and a replay-local GEMM matched its gathered rank
  slice bitwise, proving AG and rank-major restoration were not the error.
- On the unmodified real activation all four consumer segments were bitwise
  exact. Across 128 deterministic bounded mutations, however, 100 outputs
  differed on every rank: qkv/core remained exact, while index-compressor and
  index-weight differed on 94 and 25 mutations. Maximum absolute error was
  `0.03125` and maximum relative L2 was `9.2931e-5`. An explicit replay check
  confirmed the local-GEMM-to-AG slice itself was exact; the difference comes
  from changing hipBLASLt reduction/solution shape across the original small-N
  projection boundaries, consistent with the rejected TP8 output-N path.
- Seven-sample four-slot ABBA rank-max gave A means `123.559 us`, B means
  `67.669 us`, a real `55.890 us` (`45.23%`) local saving. It passes the
  predeclared `>=30 us` performance gate but fails the required 100+ mutation
  bitwise gate, so the combined-N TP4 path must not proceed to production.
  Preserving four independent output-N boundaries would be required for exact
  semantics, but prior TP8 segmented experiments show that several small local
  GEMMs erase most or all of this gain.

### TP4 M32 hybrid output-N N3584 rejection (2026-08-30)

- Extended the same standalone oracle with `--mode hybrid`: only qkv N1536
  plus core-compressor N2048 are concatenated to global N3584, each TP4 rank
  computes one full-K N896 shard and reconstructs N3584 through registered AG;
  index-compressor N512 and index-weight N64 retain their original independent
  `F.linear` calls and final concatenation order.
- The first 128-mutation gate failed decisively, so the planned 1000-mutation
  extension was not run. Even the unmodified real layer-20 activation made the
  qkv/core segments non-bitwise; across mutations qkv differed 124/128 and
  core 128/128, while both untouched index segments remained exact. Every rank
  reported the same result, maximum absolute error was `0.0078125`, maximum
  relative L2 was `3.2453e-5`, and the replay-local GEMM matched its gathered
  rank slice exactly. The discrepancy is therefore the changed N3584/N896
  hipBLASLt reduction shape, not AG or rank-major reconstruction.
- Seven-sample ABBA over the complete four-projection chain measured baseline
  A `123.380 us` and hybrid B `116.005 us`, saving only `7.375 us` (`5.98%`).
  It fails both required gates: not bitwise and far below `30 us`. Do not test
  this hybrid in production and do not spend time on a 1000-mutation rerun.

### TP4 M32 DPP gate R1/W4 with row-prefetch down rejection (2026-08-30)

- The grouped gate template launches `kNumWaves * 64` threads and maps one
  wave to one `(sorted expert block, I-row tile)` task, grid-striding by
  `gridDim * kNumWaves`. It syntactically supports W4/W8/W16; W16 is the
  hardware maximum 1024-thread workgroup. The only LDS is the 256-entry
  `uint32` FP4 pair LUT (1024 bytes), independent of wave count. On CDNA2 the
  measured R2/W8 kernel uses 96 VGPR, limiting a SIMD to two waves and a CU to
  eight waves: one W8 CTA/CU. The R1/W4 form uses about 80 VGPR, allowing three
  waves/SIMD or 12 waves/CU: up to three W4 CTAs/CU. W16 would require four
  waves/SIMD in one CTA and therefore force <=64 VGPR or spills; it is legal at
  the template/launch level but not a credible extension of this accumulator.
- Extended `scripts/rocm/bench_dsv4_tp4_m32_dpp_downprefetch_combo_oracle.py`
  with baseline DPP gate R2/W8/G2080 and candidates DPP gate R1/W4 at
  G1664/2080/2496/3120. Every profile kept the exact row-prefetch R2/W8/D832
  down path. The diverse route had 106 active experts, 192 useful assignments
  and 113 padded A4 scan blocks; with I512/R1 this is 57,856 gate row tasks, so
  W4's higher resident-wave ceiling competes against more grid-stride rounds.
- One hundred mutated inputs preserved intermediate BF16, quantized activation
  and scale, FP32 partial and final BF16 outputs bitwise for every profile.
  Seven-round forward/reverse ABBA trimmed full-stage means were baseline
  `425.374 us`, W4/G1664 `441.522`, W4/G2080 `431.333`, W4/G2496 `432.303`,
  and W4/G3120 `425.434 us`. Gate-only means were `246.888 us` baseline and
  `260.897/250.701/251.110/247.189 us` for those candidates. The best candidate
  is still `0.014%` slower, so the theoretical 12-wave/CU occupancy does not
  overcome its extra R1 row tasks. Do not add a W4 service selector or pursue
  W16; retain R2/W8/G2080 with the row-prefetch down candidate for service A/B.

### TP4 M32 serial-row DPP gate static rejection (2026-08-30)

- Added oracle-only
  `gfx90a_fp4_expert_gate_serial_rows_oracle.cuh` and a `SERIAL` loader in the
  DPP/down-prefetch combo script. It keeps the A4/R2/W8/G2080 wave/task mapping
  and the exact per-row K-group, SDOT, DPP reduction and BF16 store order, but
  completes row0 before reusing `gate_acc/up_acc[assignment]` for row1. Weight
  and activation dot calls remain one per assignment/row/group, so source-level
  bytes and global task count do not increase.
- A direct gfx90a resource build with
  `-Rpass-analysis=kernel-resource-usage` reported `99 VGPR`, `51 SGPR`,
  `1024 B LDS`, zero VGPR/SGPR spills, zero scratch and compiler occupancy
  `4 waves/SIMD`. This is worse than the `<=80 VGPR` continuation gate and does
  not deliver the expected accumulator-lifetime reduction: unrolling and
  scheduling retain weight-decode/dot temporaries across the serial row body.
- Per the predeclared rule, no 100-mutation or seven-round ABBA run was made.
  Do not connect this kernel to production. A future attempt would need an
  explicit noinline row helper or separate kernel phase to force liveness, but
  either risks call overhead or duplicated metadata/task setup and is not a
  priority after this static miss.

### TP4 M32 serial-row min-blocks=2 static rejection (2026-08-30)

- Added an independently cached compile variant of the serial-row oracle with
  `__launch_bounds__(kNumWaves * 64, 2)` for W8, using
  `SGL_SERIAL_ROWS_MIN_BLOCKS=2`. The ordinary serial-row module retains its
  default min-block setting, so the forced-residency code object cannot alias
  or overwrite its JIT cache.
- A direct gfx90a resource build reported exactly the same allocation as the
  unconstrained attempt: `99 VGPR`, `51 SGPR`, `1024 B LDS`, zero scratch,
  zero SGPR/VGPR spill and compiler occupancy `4 waves/SIMD`. The min-blocks
  hint did not shorten live ranges or force the expected <=64-VGPR allocation;
  the compiler already considers the generated code compatible with its
  occupancy model.
- This exceeds the explicit `<=64 VGPR` continuation gate. No mutation or ABBA
  test was run, and this variant must not be connected to production. Further
  launch-bound coercion is unlikely to help without a structural live-range
  split and would risk real scratch traffic.

### TP4 M32 CTA-wide activation staging gate rejection (2026-08-30)

- Added oracle-only
  `gfx90a_fp4_expert_gate_cta_stage_oracle.cuh` and `--cta-stage-only` support
  in the DPP/down-prefetch combo oracle. The CTA loop maps all eight waves to
  consecutive R2 tiles of the same sorted expert block; every thread executes
  the same CTA-task loop and both barriers. For each A4 block, 512 threads
  stage four INT8 K4096 activations and four 128-entry FP32 scale rows once,
  then the eight waves share them. Weight loads, SDOT/DPP order, BF16 stores,
  output task coverage and the row-prefetch R2/W8/D832 down path are unchanged.
- Static gfx90a resources were `91 VGPR`, `54 SGPR`, `19,504 B LDS`, zero
  scratch and zero SGPR/VGPR spills; compiler occupancy was `5 waves/SIMD`.
  The 19 KiB LDS footprint includes the 1 KiB FP4 LUT plus staged activation,
  scales and metadata. This passed the no-spill resource gate.
- One hundred mutated cases preserved intermediate BF16, quantized activation
  and scale, FP32 partial and final BF16 outputs bitwise. Seven-round
  forward/reverse ABBA nevertheless moved gate trimmed mean from `247.256` to
  `394.703 us` and full routed stage from `425.462` to `575.309 us` (35.2%
  slower). Quant/down/reduce remained essentially unchanged.
- Explicit staging loses to the cache-served peer-wave reloads: each CTA task
  adds roughly 18 KiB of global-to-LDS copies, LDS reads and two workgroup
  barriers, while the original repeated activation reads are already highly
  cacheable and small relative to FP4 weight traffic. Do not connect this
  candidate to production or pursue larger staged tiles.

### TP4 M32 per-expert readiness scheduling oracle (2026-08-30)

- Added standalone
  `gfx90a_readiness_schedule_oracle.cuh` and
  `scripts/rocm/bench_dsv4_readiness_schedule_oracle.py`. It models 113 sorted
  expert blocks with 32 producer CTAs per block. Every producer CTA performs a
  system-scope acq_rel counter RMW; the final RMW observes the release sequence
  and publishes a monotonically increasing ready epoch. A configurable
  8/16/24/32/48/64-CTA consumer device queue uses system acquire, executes
  tunable dummy work and stores a consumed epoch. Queue reset is captured in
  the consumer graph, while counters/epochs remain monotonic to avoid ABA.
- With consumer work set to 512 ALU iterations per simulated block, all six
  CTA counts completed 1000 graph replays without hang or stale data. Every
  periodic check matched `counter == epoch*32`, `ready == epoch` and
  `consumed == ready`. Synthetic producer median interference was CTA8
  `+0.071%`, CTA16 `+0.849%`, CTA24 `+1.182%`, CTA32 `+1.177%`, CTA48
  `+2.231%`, and CTA64 `+2.473%`: all below the 5% scheduling gate.
- A second independently captured graph used the real exact DPP
  A4/R2/W8/G2080 gate on the pass37/layer34 route while a pressure graph ran
  equivalent total dummy work on the alternate stream. Seven-round medians
  showed real-gate interference of CTA8 `+0.501%`, CTA16 `+0.750%`, CTA24
  `+0.125%`, CTA32 `+0.249%`, CTA48 `+0.250%`, and CTA64 `+0.312%`; no point
  approached the 5% limit.
- This establishes that the system-scope epoch protocol is graph-stable and
  that reserving 8--24 consumer CTAs need not materially delay gate. It does
  not establish the `full <=410 us` target: dummy ALU lacks real quant/down
  LDS, FP4 weight VMEM and SDOT pressure. If the full consumer is implemented,
  begin with 8/16/24 CTAs and retain the exact release sequence; do not infer a
  service speedup from this scheduling-only result.

### TP4 BS32 exact DPP-gate plus down-prefetch checkpoint (2026-08-30)

- Revisited the exact standalone combination only because the long-generation
  audit proved that the old HTTP resident metric under-reports model decode.
  Added the strictly shape-guarded
  `SGLANG_DSV4_GFX90A_M32_DPP_GATE_DOWN_PREFETCH` selector: only gfx90a
  E256/M32/T6/I512/H4096, A4/R2/W8, gate2080/down832 and LDS mode2 can select
  the DPP gate reduction plus two-row down weight/scale prefetch. The existing
  fixed-order FP32 partial reduction is unchanged. Other tiers, prefill, MFMA
  and non-TP4 shapes retain production kernels.
- The prior component oracle had already passed 100 mutated activation/router
  cases bitwise at the intermediate BF16, down FP32 partial and final BF16
  boundaries. A fresh 32-row teacher-forced service comparison found all
  output IDs, complete output-token logprobs and top-5 logprob rows bitwise
  identical between selector off/on.
- Two independent A and B services each ran three 512-token rounds with the
  same 32 distinct fixed input IDs. All twelve A/B request waves passed the
  France gate and every request returned 512 tokens. Scheduler log samples
  above 600 tok/s gave A1/A2 medians `694.945/695.895` (trimmed means
  `695.916/696.276`) and B1/B2 medians `707.990/706.140` (trimmed means
  `708.281/706.881`). This is a reproducible model-decode gain of roughly
  `1.6--1.9%`. Client common-resident samples also moved from A's
  `609.013--612.637` to B's `616.900--621.397 tok/s`; group-wall outliers remain
  a separate HTTP drain issue.
- Enable the selector by default only in the explicit TP4-BS32 profile, while
  preserving an environment override for rollback. This is below the usual
  5% checkpoint threshold but is retained because it is exact, positive in
  both independent service orderings, and follows the requested policy of
  keeping verified small wins. The new stable diverse-request model-decode
  center is approximately `707 tok/s`, still far below the 1500 tok/s goal.
### TP4 M32 G2080 DPP gate producer-release oracle (2026-08-30)

- Added standalone diagnostics only: `gfx90a_fp4_gate_producer_release_oracle.cuh`
  and `bench_dsv4_gate_producer_release_oracle.py`; no production selector.
- Preserved W8/R2/G2080 DPP arithmetic. Each CTA-uniform outer iteration
  publishes one device-scope acq_rel counter contribution; the 32nd publishes
  the ready epoch. Same-GCD cross-stream ordering only needs DEVICE scope.
- Correctness: 100 randomized mutations, intermediate BF16 bitwise exact;
  all active counters were exactly `epoch*32` and ready epochs exact.
- Seven-round ABBA trimmed result after removing redundant explicit fence and
  trailing barrier: baseline 246.496 us, release candidate 282.589 us,
  +36.093 us (+14.643%). This fails the required <10 us producer overhead.
- SYSTEM scope was much worse (246.188 -> 2555.732 us); do not use it for this
  same-GCD protocol. The fine-grained 32-contribution publication tax is too
  high even at DEVICE scope, so do not proceed to the exact consumer on this
  producer decomposition without reducing publication fan-in.
### TP4 M32 expert-owned DPP gate publication oracle (2026-08-30)

- Recorder metadata clarification: M32/top6 has 192 useful assignments but
  per-expert A4 padding expands `num_valid_ids`/`sorted_ids` to 452 entries and
  `sorted_experts` to 113 blocks. Thus the real gate grid upper bound for this
  trace is 113, not the global `ceil(192/4)=48`; the absolute worst case is 192
  padded expert blocks.
- Added standalone owner/fan-in gate oracle (no production selector). One
  expert block is processed by 1/4/8 W8 CTAs; each CTA walks its share of the
  32 R2 row rounds and publishes once with DEVICE-scope acq_rel. The last CTA
  publishes a monotonic ready epoch.
- 100 randomized mutations: all owner1/4/8 intermediate BF16 tensors are
  bitwise identical to the G2080 DPP reference; counters and epochs exact.
- Seven-round ABBA trimmed results versus G2080 reference 246.494 us:
  owner1 588.087 us (+341.593, +138.581%); owner4 322.113 us (+75.619,
  +30.678%); owner8 281.777 us (+35.283, +14.314%).
- Literal owner CTA underfills latency hiding despite 113 blocks covering the
  roughly 104 CUs. Increasing fan-in converges toward the original scheduler,
  but owner8 still fails the <10 us publication gate. Do not attach the exact
  consumer to these decompositions.

### TP4 M32 C4 attention issue-order checkpoint (2026-08-30)

- Re-captured the current HEAD layer-20 realtime marker after the accepted C4
  multistream and DPP-gate/down-prefetch changes. Across 512 hot four-rank
  groups, rank-max medians were MHC-entry `88.00 us`, prepare `154.72 us`,
  attention core `51.84 us`, output/collective `132.48 us`, FFN entry
  `95.84 us`, routed span `549.60 us`. Fine MoE medians were router `28.32`,
  top-k `12.16`, routed experts `464.48`, join/add `4.00/4.32`, and TP4 AR tail
  `41.44 us`. Marker logging reduced the HTTP resident rate to about 666 tok/s,
  so these are localization data rather than a throughput checkpoint.
- Added a default-zero issue-order selector for the narrow TP4/M32 multistream
  path. Mode 0 is the accepted schedule (both C4 compressor branches launch
  before q_lora); mode 1 delays only indexer-compressor, mode 2 delays only
  core-compressor, and mode 3 computes q_lora first then launches both. No
  tensor math, stream dependency, weights, cache layout or attention semantics
  change.
- Marker rank-max results rejected modes 1 and 2: prepare increased to
  `175.20/169.12 us`. Mode 3 reduced prepare to `149.60 us` and the following
  attention-core interval to `40.40 us`, versus `154.72/51.84 us` for mode 0,
  a combined `16.56 us` C4-layer reduction. Its q-lora projection interval was
  `44.80 us` rather than the contended baseline `68.32 us`; delayed compressor
  tail rose to `27.04 us` but remained hidden more effectively by later work.
- A 32-distinct-input teacher-forced comparison between independently started
  mode-0 and mode-3 services was JSON-exact for every output ID, complete
  logprob row and top-5 entry. Five 512-token no-marker resident rounds were
  mode 0=`620.047/620.541/619.931/621.008/619.928 tok/s` (median `620.047`,
  trimmed `620.173`) and first mode 3=`623.917/623.478/623.272/623.655/623.794`
  (median `623.655`, trimmed `623.642`). An independent second mode-3 service
  measured `621.916/626.753/623.392` (median `623.392`). The stable gain is
  about `0.52--0.58%` in the HTTP-resident metric.
- Enable issue order 3 only in `SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE`; the
  environment selector remains a zero-default rollback outside that profile.
  This is a verified small scheduling win, not a 1500 tok/s checkpoint.
### TP4 M32 static A4 sorter-only oracle (2026-08-30)

- Added standalone `gfx90a_m32_a4_sorter_oracle.cuh` and benchmark; no
  production selector. It consumes `[32,6]` expert IDs/weights and emits the
  same AIter capacities (`1210` sorted assignments, `303` expert blocks,
  two-word valid metadata). The real diverse layer34 route has 192 useful
  assignments and 452 padded entries / 113 A4 blocks.
- The atomic cursor inherited from the fused quant-sort kernel permuted 27
  same-expert assignments versus AIter. The exact oracle therefore uses a
  stable flattened token/slot rank for each assignment, matching AIter's
  deterministic order and its zero-weight padding.
- Correctness: real layer34 plus 99 randomized unique-top6 distributions were
  exact for valid sorted IDs, weights, expert IDs and num-valid metadata; 1000
  captured graph replays remained exact.
- Seven-round ABBA: AIter production sorter `10.228 us`; static stable HIP
  candidate `15.563 us` (`-5.335 us`, 1.522x slower). It fails both the >=8-us
  saving and <=60% latency gates. Do not replace `moe_sorting`; the entire sort
  is only about 10 us, so fusing top-k with sort has a small absolute ceiling.
### TP4 M32 unified-KV paged-decode core geometry oracle (2026-08-30)

- Added only oracle escape hatches for launch `num_warps/num_stages` and an
  independent benchmark; production defaults remain `(4,2)`. Fixed the actual
  TP4 decode shape T32/H16/D512, BLOCK_H16, KV_SPLITS4, BLOCK_K16 and BF16 KV.
- Swept warps 2/4/8 and launch stages 1/2/3 at uniform context lengths
  256/512/768/1024. All nine profiles were BF16 bitwise equal at each length;
  additionally, baseline `(4,2)` and candidate `(2,1)` were bitwise exact over
  100 captured Q mutations per length.
- Seven-round ABBA baselines `(4,2)` were 69.591/117.382/164.876/215.160 us.
  Keeping production stages=2 and changing only to two warps measured
  49.134/77.861/102.203/130.472 us, saving 20.458/39.522/62.673/84.688 us
  (about 29--39% lower latency). The absolute best two-warp stage choice varied
  only at sub-microsecond noise; launch stages do not materially affect this
  kernel because its inner `tl.range` already carries an explicit pipeline.
- Eight warps was 111.545/202.187/290.413/382.021 us and is decisively wrong.
  Two warps passes the >=5-us / >=10% continuation gate at every tested length.
  The next production experiment should change only `num_warps=2` for the
  exact TP4 M32/H16/D512/BF16 split-K shape, retain stages=2, run the standard
  teacher-forced bitwise check, then service ABBA. Do not generalize to other
  head counts, graph tiers, FP8 KV, or fused single-split kernels.
### TP4 M32 unified-KV two-warp service rejection (2026-08-30)

- Added a default-off, rollback-safe selector restricted to gfx90a,
  attention-TP4, T32/H16/D512, BF16 KV, split-K (`kv_splits>1`), BLOCK_H16
  and BLOCK_K16. It changes only core `num_warps=4 -> 2` and keeps stages=2.
- The 32-row teacher-forced response was bitwise identical to the adjacent
  baseline for all output IDs, complete token-logprob rows and top-5 rows.
- Five 512-token diverse-request rounds all passed France first-nine and
  returned exactly 512 tokens. Resident throughput was
  `621.821/623.186/622.691/622.048/624.003 tok/s`, median `622.691` and
  trimmed mean `622.642`, versus the current issue-order-3 baseline around
  `623.4 tok/s`. Thus the 29--39% isolated core reduction is hidden by the
  full attention multistream graph and does not improve service throughput.
- Removed the TP4 profile opt-in after the A/B. Keep the environment selector
  default-off and retain the standalone geometry oracle; do not enable it by
  default from microbenchmark results alone.
### TP4 A4/R2 two-wave K-half gate oracle rejection (2026-08-30)

- Built an independent W8-as-four-wave-pairs prototype. Wave0 processes group
  `lane`, wave1 group `lane+64`; wave1 publishes all sixteen FP32 gate/up
  partials per lane to LDS, then wave0 adds the second half before the original
  DPP reduction tree. A task-end barrier prevents next-iteration overwrite.
- Static resources passed the requested gate: 74 VGPR, 48 SGPR, 16,384 B LDS,
  private/scratch 0, VGPR spills 0 and SGPR spills 0. The initial 64-KiB hand
  estimate was incorrect: four pairs x 64 lanes x 16 floats is 16 KiB.
- Used barrier-safe grid904 for the real 113-block route: 904x4 pairs divides
  113x256 row tiles exactly, giving every pair eight uniform iterations.
- The first activation mutation nevertheless differed from the production DPP
  gate by max-abs 0.25. Adding the task-end barrier did not change the result;
  comparing against arithmetic-unpack rather than LDS-LUT production also
  retained max-abs 0.25, so LUT decoding is not the cause. The split-wave
  accumulation is not bitwise equivalent despite preserving the source-level
  intended association. Stop before ABBA and do not wire production.

#### Exact integer-dot revision and final performance rejection

- The `0.25` discrepancy above was subsequently isolated to compiler FMA
  association: wave1 had published an already scaled FP32 term.  The corrected
  oracle publishes its raw `int32` SDOT accumulator and FP32 combined scale;
  wave0 then executes the same production-shaped
  `acc += float(dot) * scale` expression for both K halves.
- This corrected form was bitwise exact for all intermediate gate/up outputs
  across 100 activation mutations.  Its static resources were 76 VGPR and 50
  SGPR by code-object metadata, 32,768 B LDS, zero private/scratch and no
  spills.  (The assembler's `next_free` values include a different aligned
  register accounting and were 97/100.)
- Seven-round ABBA measured production W4/G2080 DPP at `246.242 us` and the
  exact two-wave K-half candidate at `502.390 us`: a `256.149 us` regression,
  or about 2.04x latency.  The extra wave, 32-KiB LDS exchange and two CTA
  barriers dominate any K-parallel benefit.
- Final conclusion: K-half splitting is mathematically viable when integer
  dots and scales cross the wave boundary separately, but this CTA-cooperative
  design is decisively unsuitable for production.  Do not revisit it without
  eliminating the LDS exchange and CTA barriers altogether.

### DSpark gamma-3 M128 global A2 rejection (2026-08-31)

- Screened `SGLANG_DSV4_GFX90A_FP4_GROUPED_DECODE_ASSIGNMENTS=2` only in the
  existing gamma-3 `TARGET_VERIFY` service, using 32 heterogeneous code
  requests and the common resident BS32 window.  No code/default changed.
- Three rounds were `900.880/875.145/881.829 tok/s`, median `881.829`, about
  0.7% below the retained A4 checkpoint median near `887.837 tok/s`.
- Correctness passed in all rounds (France first-nine exact/Paris and 32x256
  `finish=length`).  Keep A4; do not change global AIter geometry or AR.

### DSpark gamma-3 anchor occupancy and quant-only rejection (2026-08-31)

- Forty real heterogeneous M128 target forwards (1,720 layer samples) show a
  mean 107.18 active experts and 63.88 singleton experts per layer.  A4 needs
  113.17 weight blocks versus 134.24 for A2; A8 only falls to 108.25.  This is
  a diffuse-routing workload: keep A4 and stop pursuing long expert runs.
- A strict target-only HIP kernel quantized only 32 anchor rows and was bitwise
  equal over 100 mutations plus 1000 graph replays.  It reduced the isolated
  quant from about 38.0 to 9.33 us, but corrected service medians were control
  901.62 versus candidate 895.65 tok/s (-0.66%); acceptance-normalized results
  also favored control.  The kernel and all production wiring were removed.
- An apparent ~488 tok/s regression during bring-up was a launch error, not a
  kernel result: the profile variable was misspelled, omitting
  `--enable-single-batch-overlap`.  Always verify the process command line.

### DSpark gamma-3 M128 CK sparse-decode rejection (2026-08-31)

- Generalizing CK/MFMA split-2 to M128 looked 39--41% faster than the oracle's
  forced Triton split-4 at contexts 128/256/512 and passed 100 mutations plus
  1000 graph replays.  That comparison was not the production geometry.
- Production M128 has enough `T*H` parallelism to select fused Triton
  `kv_splits=1`.  Layer-20 markers measured CK around 68--71 us and warm Triton
  around 68 us.  Real-code medians were control 901.625 versus CK 891.940
  tok/s; acceptance-normalized results also favored control.
- Removed the strict target-only selector and restored the wrapper's M96 cap.
  Future M128 oracles must use the production split-1 heuristic as baseline.

### DSpark gamma-3 M128 issue-order/down-quant rejections (2026-09-01)

- M128 issue order 0 had seven-round median/trimmed resident throughput
  `902.654/900.268 tok/s`, effectively equal to the mode-3 control median
  `901.625`; keep mode 3.
- Strict anchor-only intermediate quant reduced the isolated `[128,6,512]`
  group-32 quant from 38.61 to 9.65 us and was bitwise exact, but the complete
  routed stage only fell 378.330 -> 365.778 us.  Service median was 859.350,
  so all candidate wiring was removed.
- Fixed the occupancy oracle's `-1` handling: Python `buckets[-1]` had silently
  mapped dropped DSpark rows to expert255.  The helper now filters invalid IDs,
  matching production AIter sorting.

### DSpark position acceptance and lean-graph rejection (2026-09-01)

- The real32 1024-token resident-bin harness now records stream events/s and
  tokens/event.  Events stayed roughly 400--430/s while tokens/event rose from
  2.1--2.4 to 3.7--4.0; late 1.5--1.6k tok/s is acceptance/content driven, not
  kernel warmup.  Treat it as a quality-sensitive DSpark result.
- The 32K pool caused retraction/re-prefill in a long round.  A 49,152-token
  pool plus BS1/BS32-only graph did fit and retained about 1425 resident tok/s,
  but uncaptured drain tiers fell back to slow eager execution and eventually
  hit `hipErrorIllegalAddress` at eager M14.  Reject the lean graph profile;
  keep full 1--32 tiers.  This was a software path failure, not hardware.
- Full 1--32 graph capture plus the same 49,152-token pool at static-memory
  fraction 0.96 is stable: two real32 1024-token rounds completed in
  27.83/28.60 s, resident 1459.66/1442.98 tok/s, France 2/2 semantic, with no
  retraction.  Promote 49,152 and 0.96 as the TP4 BS32 defaults; this is a
  capacity/stability gain, not a kernel-speed claim.
- Rechecking `AITER_GFX90A_AR_1M_BLOCKS=12` on that profile yielded
  1477.89/1473.53 resident tok/s versus the adjacent 80-block control
  1459.66/1442.98 (+1.68% by two-round mean), but France fell from 2/2 to 1/2.
  Keep 80 blocks; the micro win is too small end-to-end and not correctness-safe.
- Detailed evidence is in
  `dsv4_dspark_position_acceptance_and_lean_graph_rejection_20260901.md`.
- **Superseded on 2026-09-01 by the combined schedule:** the single-variable
  rejection above remains valid in isolation, but composing 12-block 1-MiB AR
  with exact M32 gate-row prefetch produced a three-round 1541.26 tok/s mean
  versus 1430.37 for the full rollback (+7.75%), with France 3/3. A fresh
  no-override service centered at 1504.26 tok/s with France 3/3. The accepted
  TP4/BS32 DSpark default is therefore `AITER_GFX90A_AR_1M_BLOCKS=12` together
  with `SGLANG_DSV4_GFX90A_M32_GATE_ROW_PREFETCH=1`; native AR remains
  unchanged. See `dsv4_dspark_ar12_m32_rowprefetch_checkpoint_20260901.md`.
- M128 wave64 MHC rows/program `3/6/12/24` measured
  `60.62/59.61/81.70/116.19 us`; rows6 is exact but only 1.7% faster, so it is
  too small for a service change.
- `SGLANG_DSV4_GFX90A_M32_GATE_ROW_PREFETCH=1` on the accepted 49K profile
  produced resident `1472.94/1488.54` versus disabled rollback
  `1427.45/1517.03 tok/s`.  Both were France 2/2; the two-round mean gain is
  only 0.58%, within acceptance noise.  Keep it disabled.
- A strict DSpark M128 compact-routed in-place anchor add was 100/100 bitwise
  exact, but its captured zero/scatter/add chain improved only
  `12.23 -> 8.70 us/layer`.  This is below 0.2% of the full step; remove the
  prototype without spending a service launch.
- Current 49K gamma3/M128 layer-20 rank-max markers put the full-tier layer at
  about `1.43 ms`; fine MoE is router `32.6--34.2 us`, top-k `12.5--12.8 us`,
  compact M32 routed `453.1--459.0 us`, join/add `8.5--9.1 us`, and TP4 AR
  tail `73.3--80.2 us`.  Marker logging depresses HTTP throughput and is only
  localization evidence.  The remaining ~60-us/layer target budget must come
  from a new routed-stage work decomposition, not accumulated small switches.
- Exact R1/W4 gate/up phase fission was 100-mutation and 1000-graph-replay
  bitwise exact, but split grids 1664/2080/2496/3120 took
  `405.38/391.33/381.52/375.13 us` versus the combined R2/W8 gate at
  `246.07 us`.  Two phase launches and doubled R1 task setup overwhelm the
  lower accumulator pressure; remove the oracle.
- Strict DSpark M128 learned-router Top5 (hash layers remain Top6) failed
  France 0/3 and produced only `1219.75/1166.01/1179.89 tok/s`, with mean
  acceptance `2.87--3.01`.  It harms both quality and acceptance; all wiring
  was removed.  Do not prune target-anchor experts further.
- Official full-block draft attention was rechecked after the CPU-length and
  live-`swa_loc` fixes.  Gamma3/49K real32 1024-token rounds were
  `1413.26/1479.45 tok/s` (mean 1446.36), acceptance `3.560/3.634`, and France
  0/2, versus control mean 1451.32 and France 2/2.  Keep it disabled; the next
  acceptance investigation must compare fixed-input draft graph/eager states.
- Disabling only the gamma3 draft CUDA graph gave `1441.23/1391.25 tok/s`,
  acceptance `3.664/3.578`, and France 1/2.  Its 1416.24 mean is below the
  graph control; the gamma5 graph/eager issue does not explain the current
  gamma3 gap.  Keep the draft graph enabled.
- Standalone NGRAM gamma3/breadth1 with the same TP4/49K target reached only
  `568.47 tok/s`, acceptance 1.803, and failed France in its first 32x1024
  real-code round.  Full M128 routed verification overwhelms sparse code
  matches; stop before a second 70-second round and retain DSpark.
- Gamma3 `scheduler_recv_interval=2` gave a very stable
  `1428.594/1428.597 tok/s`, acceptance `3.533/3.512`, and France 2/2, but is
  1.6% below the interval-one control mean.  Remove the temporary launcher
  passthrough; scheduler polling is not the missing 3%.
- A compact gamma3 M96 graph with 32 anchors plus the 64 highest-confidence
  draft rows, and a strict ragged anchor-only routed mask, reached only
  `992.34 tok/s`, acceptance 2.590, in the first real32 256-token round.
  France remained Paris, but graph savings did not offset lost drafts and
  M96 router/top-k work.  Remove all temporary M96 wiring.
