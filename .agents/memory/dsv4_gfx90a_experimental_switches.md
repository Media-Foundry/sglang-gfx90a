# DSV4 gfx90a 实验开关清单

范围：本地 bring-up/优化提交 `505b337379`、`2a98bfbb1f` 以及
`scripts/rocm_dsv4_flash.sh`。这里不把目标仓库的普通上游开关当成我们新增的
调试开关。当前测试口径仍是 TP4/EP4、Mori、batch=1、原生 AR。

## GPU 实验前置检查（强制）

每次启动性能 probe、服务 A/B 或 profiler 之前，先运行：

```bash
amd-smi process --general --sort-by-pid -g 4 5 6 7
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
