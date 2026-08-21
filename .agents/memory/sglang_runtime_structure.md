# SGLang 推理结构（人工浅扫）

## 请求到 GPU forward

```text
sglang serve / python -m sglang.launch_server
  -> srt.entrypoints.http_server.launch_server
  -> Engine / TokenizerManager + Scheduler 子进程 + DetokenizerManager
  -> Scheduler.run_batch
  -> ModelWorker.forward_batch_generation
  -> ModelRunner.forward / _forward_raw
  -> eager forward 或 DecodeCudaGraphRunner replay
  -> logits processor / sampler / detokenizer
```

HTTP、TokenizerManager 和 Engine 在主进程；Scheduler、Detokenizer 是独立进程，
通过 IPC/ZMQ 传递批次和 token。调度、KV admission、batch mode 和通信计划在
`python/sglang/srt/managers/`；设备执行和 graph 在
`python/sglang/srt/model_executor/`。

## DeepSeek-V4 模型路径

```text
embed -> 43 decoder layers
       -> mHC pre/RMS + mix/sinkhorn
       -> DSV4 MLA/DSA attention + compressor/indexer + paged KV
       -> mHC post/pre (可跨层融合)
       -> router top-k
       -> Mori EP dispatch
       -> AIter FP4 routed MoE
          + shared expert（TP1 或 gfx90a Mori TP4 side-stream）
       -> Mori combine / shared partial all-reduce
       -> residual / 下一层
  -> hc_head + norm -> vocab-parallel lm_head -> logits processor
```

主要实现边界：

- `python/sglang/srt/models/deepseek_v4.py`：mHC、DSV4 attention、43 层循环、
  Mori shared-TP 拼接和最终 head。
- `python/sglang/srt/models/deepseek_v2.py`：通用 MLA、router、AIter MoE、
  shared expert、dual-stream/TP 组合；V4 复用其 MoE 基类。
- `python/sglang/srt/layers/attention/dsv4/`：compressor、C4/C128 indexer、
  metadata 和 sparse prefill 逻辑。
- `python/sglang/srt/layers/moe/moe_runner/aiter.py`：AIter routed/shared GEMM
  与量化；`token_dispatcher/moriep.py`：Mori dispatch/combine、dtype、capacity、
  stream 和 launch geometry。
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`：decode
  capture bucket、静态 buffer、graph replay；BS=1 时仍是 graph replay，不代表
  speculative。

## 当前 gfx90a 关注的临界边界

1. `DeepseekV4DecoderLayer._run_moe_ffn_dp_sync` 中，attention 输出先按 TP
   chunk 切分；shared TP 可在 side stream 计算，主 stream 运行 routed MoE，随后
   通过事件等待和一次 TP all-reduce 合成。这是 graph/cross-stream 的最高风险点。
2. Mori dispatch/combine 是 EP4 的跨 rank 边界；capacity、external input buffer、
   combine geometry 会改变 graph 形状和通信路径。
3. mHC custom pre-mix 当前被代码限制在 `moe_a2a_backend.is_none()`，所以不能把
   no-A2A 的 MHC 结果直接外推到 Mori。
4. `ModelRunner.forward` 外层同时承载 profiling、expert/indexer capture、
   elastic EP 和 graph/eager 选择；性能计时应优先取 server decode window，
   再按 layer/stream 拆 critical path。

## 目前不应混淆的路径

- SGLang 的通用 `speculative/`、DSpark、EAGLE 代码仍存在，因为它们是上游功能；
  `scripts/rocm_dsv4_flash.sh` 通过拒绝非零 `SPECULATIVE_*` 将本轮验收锁为原生 AR。
- prefill/TBO/CP、decode graph replay、Mori AsyncLL/SDMA 是不同 execution mode；
  它们的吞吐和同步代价不能合并成一个数字。
