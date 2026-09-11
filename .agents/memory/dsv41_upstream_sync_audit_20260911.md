# DeepSeek-V4.1 upstream sync audit (2026-09-11)

## Baseline and safety boundary

- Working branch: `sync/media-foundry-gfx90a`
- Baseline commit: `8de5b9eac40b5ed6513e159bbff782950e787177`
- Immutable local checkpoint: `backup/pre-upstream-v41-sync-full-20260911`
- Production DeepSeek-V4/gfx90a code remains the source of truth.  This sync is
  selective; it is not a wholesale merge of `origin/main` or `origin/dsv4.1`.
- The checkout has unrelated user experiments and a generated
  `cuda_graph_runner_memory_usage.pickle`; neither is reset or cleaned.

## Repository topology

`origin/main` and the V4.1 feature branches share only an old ancestor with the
gfx90a fork.  A fast-forward is therefore impossible.  Applying the V4.1 Day-0
commit directly to the baseline produced 32 conflicts across model, attention,
pool, scheduler, quantization, and speculative code.  This is expected and is
why new V4.1 files are imported behind an isolated registration boundary first.

Audited refs:

| Ref | Purpose | Decision |
|---|---|---|
| `origin/main` (`94ce940ff8`) | current upstream master | audit individual fixes only |
| `origin/dsv4.1` (`3b709e55c0`) | complete V4.1 feature line | source for isolated new modules |
| `origin/refactor/dsv41-multimodal-layout` (`6407cc5197`) | V4.1 multimodal follow-up | defer shared refactor; inspect later |
| `origin/codex/dsv41-indexer-plan` (`cfcc57c411`) | candidate-indexer experiments | reference only until local contracts pass |

## Upstream commits worth reviewing

### V4.1 support

- `85f8105f5d` Day-0 V4.1 support: introduces the model/config, Engram,
  ratio-1/2 compressor/indexer, encoding, vision, and broad shared plumbing.
  **Do not cherry-pick wholesale**; it changes ~150 files and conflicts with
  local gfx90a kernels.
- `b1ad23f677`, `dc5f59c3a2`, `41da06adca`: config/metadata and compressed-pool
  refactors needed by the full upstream model, but they touch the current V4
  pool and backend.  Port only after a V4.1-specific adapter and tests exist.
- `1aa0e962b2`, `7bdebdab7d`, `f3c3e7c7ad`, `33e847d080`, `824bb45e1b`,
  `e087e662ba`, `3b709e55c0`, `6407cc5197`: later V4.1 correctness/API fixes;
  apply selectively after the isolated model imports cleanly.
- `c36636b7da`, `759baff47f`: Blackwell-specific DSpark/MoE optimizations;
  **not applicable to gfx90a** and must not replace the local HIP path.

### AMD/DSA fixes in upstream master

These are candidates for separate, one-commit ABBA validation, not part of the
initial V4.1 import:

- `429ac2d82c`: draft worker must use the draft compression path during prefill.
  The local file already has part of this guard; compare before porting.
- `92d831d3d7`: token-block parallel KV-index construction for long-context
  speculative decode.  Default remains historical for short contexts upstream;
  port only if local AIter contracts match.
- `94ce940ff8`: include DSA variant in exact-bucket graph admission.  Local
  graph-variant code is newer/different; verify the equivalent is already
  present before changing it.
- `8656901504`: avoid specializing the prefill page-table stride; likely useful
  for JIT-shape stability, but it targets the generic DSA transform and needs a
  local compile/test.
- `15aa2fb843`: fused DSA metadata and redundant absorb work.  Potentially
  useful on HIP, but it changes shared ROCm attention code and must be isolated.
- `31ebd8f437`: fused FP4 indexer prefill schedule preamble.  Promising for long
  prefill, but its schedule buffer ABI must not be mixed with the current local
  indexer workspace without an oracle.
- `6cee9285a3`, `9978aaec8b`, `7ed29eba80`, `ccfa120dae`: exact Top-K,
  logits-budget, FP4 indexer bounds, and compression-tail fixes; inspect for
  already-present local equivalents and port tests first.

## Initial implementation order

1. Keep the baseline checkpoint and this audit in history.
2. Import pure V4.1-only files (config, ratio-1/2 sparse helpers, Engram,
   host-window/multimodal/encoding support) without editing the current V4
   model/backend.
3. Register `deepseek_v41` and architecture aliases, then run config/registry,
   Python syntax, and CPU/meta tests.
4. Add an isolated V4.1 model entry point.  Resolve shared API gaps with small
   adapters rather than replacing `deepseek_v4.py`.
5. Run the complete checkpoint audit and bounded Engram host smoke.  Full Engram
   tables stay in host RAM/page cache; only bounded requested rows may reach HBM.
6. Before any GPU load, record `amd-smi process --json`; only then attempt a
   real model load on an explicitly chosen GCD set.

## Correctness and rollback gates

- Existing V4/gfx90a tests and native AR/DSpark paths must remain unchanged.
- Every imported V4.1 batch gets `py_compile`, config/registry smoke, and
  teacher-forced or meta checks before a GPU run.
- Any shared-file port must be a separate commit with an explicit rollback
  point.  No upstream optimization is considered accepted from source review
  alone; it needs an end-to-end A/B/A result and an output/correctness check.

