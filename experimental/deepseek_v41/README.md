# DeepSeek-V4.1 experimental bring-up

This directory is an intentionally isolated bring-up boundary for
`DeepseekV41ForCausalLM`.  It is **not** registered in SGLang's model registry
and does not change the existing DeepSeek-V4 implementation.

## What is implemented now

* `metadata.py` reads `config.json`, `model.safetensors.index.json`, and only
  safetensors headers.  It validates the 40-layer backbone + 3 MTP layers,
  384 routed experts, compression schedule, indexer/Engram manifests, and
  TP divisibility.  Missing or `.incomplete` shards are reported as
  `incomplete`, never treated as zero tensors.  The audit also reports the
  observed CSA2 parameter layouts and an explicitly marked, metadata-only CED
  split candidate (the released V4.1 README describes 20 encoder + 20 decoder
  layers; the JSON does not expose a dispatch enum).
* `meta_model.py` builds a representative `torch.device("meta")` graph.  It
  allocates no model payload and exposes packed FP4 routed-weight shapes, layer
  ownership, and a shape-checking `forward` smoke path.
* `engram_host.py` provides a raw-byte, read-only mmap row source for Engram
  tensors, converter-compatible dim-0 row mapping, a bounded CPU staging pool,
  and a single-worker asynchronous prefetcher.  The full table remains in
  host RAM/page cache; only explicitly requested rows can be copied to HBM.

The host table deliberately preserves on-disk bytes.  Only `embed.weight` and
`embed.scale` are row-addressed by the hash id; `q_weight`, `k_weight`, and
`wkv.*` are static operator tensors and are fetched through a separate explicit
API.  FP8/scale decoding and integration into a V4.1 attention block are
deferred until shards 47 and 48 are complete and their real headers/payloads
have been checked.

## Smoke commands

From the repository root:

```bash
python scripts/rocm/audit_deepseek_v41_checkpoint.py \
  --model-dir /media/PM983/deepseek-v4.1-flash \
  --tp-size 8 \
  --json-out /tmp/dsv41-audit.json

PYTHONPATH=. python - <<'PY'
import torch
from experimental.deepseek_v41.meta_model import DeepSeekV41MetaModel

model = DeepSeekV41MetaModel.from_json(
    "/media/PM983/deepseek-v4.1-flash/config.json", tp_size=8
)
print(model.structural_manifest())
print(model(torch.zeros((2, 8), dtype=torch.long, device="meta")).shape)
PY
```

The audit exits zero for a valid but incomplete download.  Add
`--require-complete` for a CI/loader gate once all 48 shards exist.

## Engram design boundary

`EngramHostTable.from_safetensors_index(...)` does not load payloads into a
Torch GPU tensor.  `SafetensorsRowStore` maps a tensor's byte range lazily;
`EngramPrefetcher` reads selected logical rows into a small staging slot.  The
staging pool checks `RLIMIT_MEMLOCK` and falls back to pageable memory when the
machine cannot pin it.  This is intentional: the observed Engram tables are
hundreds of millions of rows, while the host's memlock limit is only on the
order of a few hundred MiB.

The first production integration should therefore be:

1. Keep all Engram rows in NUMA-aware host RAM (or the OS page cache).
2. Prefetch the union of rows needed by the next prefill/decode microbatch.
3. Copy only that bounded batch to a runner-owned GPU staging buffer; load
   static q/k/wkv tensors separately and never index them with hash row ids.
4. Retain a small optional hot-row cache only after hit-rate measurements.
5. Compare raw row bytes and reference outputs before enabling any approximate
   eviction or lookahead policy.

No CUDA/HIP graph capture should contain a page fault or an unbounded host
read.  The future runner must make the host read, device copy, stream, and
lifetime explicit before capture.
