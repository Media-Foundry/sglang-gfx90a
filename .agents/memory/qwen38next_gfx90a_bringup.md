# Qwen3.8-Flash-Next / Qwen4Exp gfx90a bring-up

Date: 2026-08-27

## Checkpoint structure

- Path: `/media/PM983/qwen3.8next`
- Architecture: `Qwen4ExpForConditionalGeneration` (`qwen4_exp`)
- 48 layers in a 3 Gated-DeltaNet + 1 QSA pattern (12 QSA layers)
- QSA: 4 query heads, 1 KV head, head dimension 128, compression ratio 4,
  token budget 2048 (512 compressed blocks)
- MoE: 512 routed experts, top-k 10, hidden size 2560, expert intermediate
  size 640
- Main + PLE + MTP safetensors total: 172.762 GiB

Exact checkpoint component sizes from safetensors headers:

| Component | GiB |
|---|---:|
| Routed experts | 112.514 |
| PLE ngram | 47.684 |
| Linear attention | 3.886 |
| MTP | 2.513 |
| Embedding / LM head | 2.368 |
| Hyperconnection | 1.193 |
| Full attention | 1.113 |
| Vision | 0.836 |
| Shared experts | 0.439 |
| Other | 0.179 |
| QSA indexer | 0.037 |

## Parallelism and quantization conclusion

TP8 is not required merely to fit this checkpoint. Native checkpoint FP8 with
TP4 + EP4 loaded successfully on four MI250 GCDs. Runtime model weight usage was
43.80 GiB/GCD. With a 0.80 static-memory fraction, the server allocated a
279,168-token BF16 KV pool and retained approximately 12.6 GiB/GCD free.

Pure TP4 + EP1 is structurally problematic for the current block-FP8 MoE path:
the expert intermediate shard is 640 / 4 = 160, which is not divisible by the
128-column FP8 block. TP4 + EP4 retains full 640-wide local experts and is the
appropriate first correctness configuration.

SGLang has MXFP4/NVFP4 support but no quantization format literally named MQ4.
The existing gfx90a custom FP4 kernels target DeepSeek-V4 shapes
(H=4096, I=256/512, top-k=6), not this checkpoint's H=2560, I=640, top-k=10.
Do not mix a new routed-expert-only MXFP4 conversion with initial correctness.
It remains a later memory/performance experiment after an FP8 oracle is stable.

## ROCm compatibility fixes

The upstream Qwen3.8 support assumed several CUDA-only JIT kernels. The initial
gfx90a correctness profile disables or replaces CUDA-only QSA top-k, MQA,
hyperconnection, PLE and grouped-RMS fusions. QSA decode also assumed
`flash_attn`/FA4 after compacting selected KV rows. On HIP, retain the existing
QSA selection and compact extraction, then use an eager per-request FP32
grouped-query attention fallback. This is a correctness oracle, not the final
performance kernel; replace it with a CK-style packed sparse HIP kernel later.

## Validated launch and correctness

Configuration:

```text
HIP_VISIBLE_DEVICES=0,1,2,3
TP=4 EP=4 MOE_A2A=none
attention_backend=aiter
context_length=4096
chunked_prefill_size=1024
mem_fraction_static=0.80
CUDA graph disabled
radix cache disabled
SGLANG_ENABLE_QWEN4_PLE_FUSION=0
```

The model loaded all 131 shards in 47.58 seconds. Two independent OpenAI chat
requests with thinking disabled, temperature 0, and the prompt "What is the
capital of France? Answer in one short sentence." both returned exactly:

```text
The capital of France is Paris.
```

Both completed with 8 generated tokens and the model stop token. Local curl
must use `--noproxy '*'` because the host's `http_proxy` otherwise routes even
127.0.0.1 through port 7897 and produces a misleading empty 502 response.

