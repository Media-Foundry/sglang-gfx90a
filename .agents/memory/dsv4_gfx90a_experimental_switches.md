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
