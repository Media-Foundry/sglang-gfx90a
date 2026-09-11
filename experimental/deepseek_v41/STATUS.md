# DeepSeek-V4.1 bring-up status (2026-09-11)

This is a checkpoint for the initial five-step bring-up.  It is intentionally
separate from the production DeepSeek-V4 path.

## Snapshot

The ModelScope download at `/media/PM983/deepseek-v4.1-flash` is still active.
The index describes 48 safetensors shards, 96,085 tensors and a reported
payload size of 510,286,023,000 bytes.  At the latest read-only audit there
were no missing shard names, but three shards were still `.incomplete`; counts
are expected to change while the downloader runs.

The manifest already establishes:

* `DeepseekV41ForCausalLM`, 40 backbone layers plus 3 MTP layers;
* 384 routed experts, top-6 routing (divisible by TP8);
* compression schedule length 43, with V4.1 ratios 0/1/2 (not the V4 4/128 schedule);
* index source layers `[2, 8, 14, 20, 24, 28, 32, 36]` and KV source layers
  `[2, 8, 14, 20]`;
* Engram at layers 1 and 14, six tensors per layer, mapped exclusively to
  shards 47 and 48;
* three MTP blocks and a vision subgraph in the same checkpoint.

The configured Engram tables have 384,006,168 and 384,016,682 rows with
256-wide FP8 rows.  The layer-14 complete header additionally confirms:
`embed.scale` is FP8 E8M0 with 8 values per row, q/k are BF16 `[4, 5120]`,
and `wkv.weight`/`wkv.scale` are static FP8 tensors `[25600, 6144]`/[800,
192].  They must stay host-resident; the current host `RLIMIT_MEMLOCK` is only
about 126 MiB, so pinning the full tables is neither possible nor intended.

## Implemented

1. `metadata.py` provides dependency-light config projection, safetensors header
   parsing, shard completeness checks, tensor/key grouping, expert divisibility,
   Engram manifest checks, and converter-compatible TP row ranges.  It never
   reads tensor payloads.
2. `meta_model.py` creates a representative `torch.device("meta")` graph and
   validates a shape-only forward.  It does not expose an `EntryClass` and is
   not registered by SGLang.
3. `engram_host.py` provides lazy safetensors mmap row stores, exact dim-0 row
   mapping (ceil-and-pad semantics), a bounded pinned/pageable CPU staging pool,
   and an asynchronous host prefetcher.  Device copies are explicit and no GPU
   table is allocated implicitly.
4. `scripts/rocm/audit_deepseek_v41_checkpoint.py` is the reproducible CLI.  It
   supports `--meta-smoke`, `--engram-host-smoke`, `--require-engram`, and
   `--require-complete`.
5. CPU-only tests are in
   `test/registered/unit/test_deepseek_v41_bringup.py` (5 tests passing in the
   DS conda environment).

## Pending gates

* Do not build a real V4.1 model or convert weights until shards 47 and 48 are
  complete and their headers pass the audit.
* Once complete, compare all six Engram tensor dtypes/shapes and raw row bytes
  against the index/converter; then construct row stores for both layers.
* Only after that integrate an explicit V4.1 model class and run teacher-forced
  CPU/reference checks.  Before any GPU run, record `amd-smi process --json`.
* Keep the host read, staging lease, HIP stream and device lifetime outside CUDA
  graph capture until a bounded, graph-safe protocol is demonstrated.

Run the following after the downloader exits:

```bash
python scripts/rocm/audit_deepseek_v41_checkpoint.py \
  --model-dir /media/PM983/deepseek-v4.1-flash \
  --tp-size 8 --require-complete --require-engram --meta-smoke \
  --json-out /tmp/dsv41-complete-audit.json
```

A non-zero result is expected until every indexed shard is present and valid.
