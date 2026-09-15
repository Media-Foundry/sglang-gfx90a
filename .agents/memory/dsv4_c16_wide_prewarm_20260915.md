# Wide-C4 compile-only priming: cross-process cache proof

Progress on drift/C16 optimization. Accepted production throughput stays
8384.698465 input tok/s; optional producer stack8692.442097 remains default-off.
No model, attention selection, KV length, dtype or production dispatch default
changed. This turn addresses first-use compilation, NOT steady throughput.

## Identified remaining cold signatures

Current query16 kernel already marks M do_not_specialize. W, page columns NP
and page stride PT remain constexpr. Thus odd M reuse is solved but new
width/layout still compiles. Prior16K/32K service logs recorded concurrent
per-rank~8.4s first-width compiles; never sum those eight rank times.

Inspected installed Triton JITFunction.warmup/run: warmup uses MockTensor
arguments and warmup=True; GPU launch is guarded by not warmup. MockTensor
assumes16-byte pointer alignment. Priming must match actual pointer alignment,
dtype, cache shuffle, dot dtype, W/NP/PT, flags/compiler/architecture/cache.
It does not magically cover all possible strides or other model kernels.

## Isolated GCD4 proof with fresh private caches

Three subprocesses: compile-only prime, fresh-process real warm replay against
that cache, and fresh-cache cold real replay. No service or other GPU test
overlapped. Prime M32768; actual M8192 and32767. No dummy KV computation was
launched during prime. A Triton launch hook observed zero launches and Torch
memory_allocated delta stayed0; this is not a claim of zero HIP-context VRAM.

| C4 width / page stride | Real M | Cold first call | Primed first call |
|---|---:|---:|---:|
|4096 /64|8192|7.425210s|0.015435s|
|8192 /128|32767|7.574118s|0.070324s|
|8192 /131|32767|7.999052s|0.076240s|

Initial compile-only calls themselves cost8.403252/8.157401/8.108582s. The work
is moved ahead of serving, not eliminated. Each cold/warm first-call number
is one observation, not an ABBA throughput statistic. Three repeated calls
afterward were approximately8.4ms,66.5ms,66.1ms, respectively.

All three prime/warm/cold cache hashes AND HSACO SHA256s agree. Warm versus
cold hashes of Q, packed cache, weights, true lengths and page table agree;
complete FP32 logits output hashes agree. Outputs finite and invalid tails
zero. Fixtures are synthetic, not live service inputs. Original runtime-M
kernel source SHA is identical across processes. Post-load compiler metadata
is64 registers,0 spills,8KiB LDS. M32768 priming really covers32767 with the
same binary. Padded stride131 is a distinct compiled specialization.

## Exported utility and tests

New helper `gfx90a_indexer_prewarm.py::prewarm_query_reuse4` returns an
unlaunched CompiledKernel. Explicit CLI:

```bash
HIP_VISIBLE_DEVICES=4 /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/prewarm_dsv4_indexer.py \
  --signature 4096:64:64 --signature 8192:128:128 \
  --output /tmp/dsv4-wide-prewarm.json
```

Output must not already exist. Default FP8=e4m3fn, dot=BF16, shuffle16 match
the audited current profile. CLI supports explicit fnuz/dot/shuffle variants,
but those variants are not newly tested here. Use the SAME TRITON_CACHE_DIR
(or same default cache), environment/compiler and runtime metadata as workers.
Signature format is WIDTH:PAGE_COLUMNS:PAGE_STRIDE. This command does not
enable wide-query or prime main attention/MoE/all model shapes.

Twelve CPU signature-validation tests passed (one existing unknown
asyncio_mode pytest warning). Public CLI reproduced all three expected binary
hashes. A separate test of the exported helper also verified all three artifact
identities, zero launches and zero Torch tensor allocation. The public helper
was not merely assumed equivalent to the prototype.

## Retained harness failure

First prototype tried reading CompiledKernel.n_regs before executable handle
initialization. Compile-only kernels do not expose it then, so the probe raised
AttributeError after compiling the first shape. This was my metadata-observation
error, not a GPU/kernel fault. Preserved failed_probe.py and prime.log/cache.
Corrected optional resource fields toNone before load and reran all phases in
a separate fresh v2 cache. Do not treat failed phase as successful evidence.

## Boundaries and next gate

No actual8-rank cold-service TTFT was measured this turn. No claim that first
real model request now has warm TTFT: other kernels/modules, projections and
metadata may still initialize. Wide-query stays default-off until actual
new-service width/page-stride/artifact hits and long/prefix behavior are checked.
The next real-service test should reuse existing16K/32K input manifests and
compare runtime compile-shape hashes to the primed set, not invent new text or
silently truncate context. Larger histories beyond8192 C4 keys are unsupported.

All subprocesses exited; final amd-smi shows no processes on all8GCDs.
No production cache was deleted or overwritten. Private experiment cache
directories are retained locally but omitted from evidence archive; sources,
compile parameters and binary hashes suffice to rebuild.

Evidence directory `.agents/experiments/dsv4_c16_wide_prewarm_20260915/`:
probe/run/analyze/helper_check sources; retained failure source/log; v2 phase
logs/results; utility.json; helper-check.json; analysis.json. Archive identity
is in evidence.sha256. Persistent goal remains active.
