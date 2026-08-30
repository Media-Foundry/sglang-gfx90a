# DSV4 TP4 M64 active-expert packed-byte roofline (2026-08-30)

## Question

How much of the accepted ~701 us/layer routed-expert interval can possibly be
removed by sorter/work-decomposition changes if every active expert's original
packed FP4 bytes still have to be fetched by one TP4 rank?

This is a diagnostic, not a production kernel.  No production selector or
model math was changed.

## Authoritative input and exact TP4 byte mapping

- Route snapshot: `/tmp/dsv4_tp4_m64_real_route.pt`
- Snapshot metadata: layer 34, pass 20, `topk_ids=[64,6]`
- Active experts: 166
- A4 scans reconstructed as `sum_e ceil(occupancy_e / 4)`: **182**
- The previously quoted 174-scan histogram is not this file.  The saved tensor
  is authoritative for this run.
- Raw checkpoint tensors are packed `int8` (two FP4 values/byte):
  - `w1=[2048,2048]`, 4 MiB/expert
  - `w3=[2048,2048]`, 4 MiB/expert
  - `w2=[4096,1024]`, 4 MiB/expert
- TP4 rank 0 uses the same sharding rule as the loader:
  - w1/w3 output rows `[0:512]`: 1 MiB each
  - w2 packed-input columns `[0:256]`: 1 MiB
  - total **3 MiB/expert**, **768 MiB for 256 experts**

The 768 MiB tensor is loaded from safetensors once, copied to GPU 6 once, and
then remains GPU-resident.  H2D and metadata copies are outside timed events.
Every measured pass reads 438--546 MiB, far larger than cache.  Cold tests
rotate expert addresses in the 768 MiB resident allocation; warm tests replay
the same addresses.

## Reader and correctness

New diagnostic files only:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_packed_weight_roofline.cuh`
- `python/sglang/kernels/ops/debug/gfx90a_packed_weight_roofline.py`
- `scripts/rocm/bench_dsv4_tp4_m64_packed_weight_roofline.py`

The HIP kernel streams 64 KiB chunks with 2,080 CTAs.  Each CTA reduces four
wave64 XOR checksums in LDS and writes one private checksum word.  There is no
global atomic contention in the timed reader.  The host XORs those 2,080 words
after timing.

Validation:

- CPU/GPU checksum exact for 100 expert-address rotation mutations.
- 1,000 HIP Graph replays bitwise stable.
- GPU process scan was clean before both runs; only physical GPU 6 was used.

## Nine-run results (discard min/max, mean of the middle seven)

| case | bytes | trimmed us | median us | effective GB/s |
|---|---:|---:|---:|---:|
| contiguous ideal, 166, cold rotation | 498 MiB | 512.09 | 511.68 | 1019.7 |
| real unique IDs, 166, cold rotation | 498 MiB | 516.91 | 514.40 | 1010.2 |
| real unique IDs, 166, same-address warm | 498 MiB | 511.31 | 512.48 | 1021.3 |
| real A4 scan pattern, 182, cold rotation | 546 MiB | 521.60 | 522.40 | 1097.6 |
| real A4 scan pattern, 182, same-address warm | 546 MiB | 523.40 | 523.68 | 1093.8 |
| historical favorable active-count floor, 146, cold rotation | 438 MiB | 466.44 | 468.32 | 984.6 |

Raw JSON: `/tmp/dsv4_tp4_m64_packed_weight_roofline.json`.

The real-ID ordering costs only ~4.8 us versus contiguous ordering.  Repeated
A4 scans have some within-launch cache reuse (therefore >1.0 TB/s when repeated
bytes are counted), but still take ~522 us.  Same-address warm replay is not
materially faster because each pass is hundreds of MiB.

## 1,500 tok/s budget

Use the accepted scheduler range 1,014--1,016 tok/s; the midpoint 1,015 gives:

```text
current M64 step       = 64 / 1015 = 63.054 ms
current mean/layer     = 63.054 / 43 = 1,466.4 us
target M64 step        = 64 / 1500 = 42.667 ms
required saving/layer  = (63.054 - 42.667) / 43 = 474.1 us
accepted routed stage  = ~701 us/layer
target routed budget if routed alone pays all saving = 226.9 us/layer
```

That routed budget is below even a read-only packed-byte pass:

- measured real unique byte floor: 516.9 us -> impossible by 290.0 us/layer;
- measured real A4 reader: 521.6 us -> impossible by 294.7 us/layer;
- favorable 146-active reader: 466.4 us -> impossible by 239.6 us/layer.

Even granting the MI250 GCD's nominal 1.6 TB/s and **zero** decode, scale,
activation, reduction, or launch cost:

```text
166 experts × 3 MiB / 1.6 TB/s = 326.4 us/layer
146 experts × 3 MiB / 1.6 TB/s = 287.0 us/layer
```

Replacing the whole 701 us routed stage with those impossible pure-read floors
would cap the current service at approximately:

| floor substituted for routed stage | aggregate upper bound |
|---|---:|
| measured real unique, 516.9 us | 1,161 tok/s |
| measured A4 pattern, 521.6 us | 1,156 tok/s |
| measured favorable-146, 466.4 us | 1,208 tok/s |
| nominal 1.6 TB/s, 166 active, 326.4 us | 1,363 tok/s |
| nominal 1.6 TB/s, 146 active, 287.0 us | 1,414 tok/s |

## Decision

Sorter/task-count work can still recover overhead and improve the ~1,015 tok/s
checkpoint, but it cannot produce 1,500 tok/s while every active expert's full
3 MiB TP4 shard is fetched every layer.  The target requires at least one of:

1. real cross-token/expert weight reuse (persistent/hot cache or a layout that
   avoids refetching the full shard),
2. fewer active experts/bytes without changing model semantics,
3. overlap that removes a separate non-routed critical-path interval, and/or
4. large attention/MHC/shared-expert savings in addition to routed savings.

In particular, A1/R8+A2/R4+A4/R2 remains useful as an overhead experiment, but
its success must not be extrapolated past this packed-byte floor.
