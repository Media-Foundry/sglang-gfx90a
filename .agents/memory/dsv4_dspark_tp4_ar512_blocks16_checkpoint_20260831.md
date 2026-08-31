# DSV4 TP4 DSpark 512-KiB AIter AR blocks=16 checkpoint (2026-08-31)

## Scope

- Original DeepSeek-V4-Flash weights, TP4/EP1/no-A2A, DSpark gamma one.
- Physical `HIP_VISIBLE_DEVICES=4,5,6,7`.
- Target workload: 32 distinct concrete coding prompts, 256 generated tokens.
- Correctness: France first-nine-token oracle after every service start; all
  requests completed at the requested length.

## Untuned AIter geometry

For BF16 `[64,4096]`, the payload is exactly 512 KiB. AIter's optimized custom
all-reduce chooses the two-stage TP4 kernel and the generic formula launches
64 CTAs x 512 threads. This is one pack per lane per stage and had not been
tuned on gfx90a.

The default-off patch
`scripts/rocm/patches/aiter_gfx90a_ar_512k_blocks.patch` exposes
`AITER_GFX90A_AR_512K_BLOCKS` only when all of the following hold:

```text
arch == gfx90a
world_size == 4
payload bytes == 512 * 1024
two-stage custom all-reduce selected
```

Every other AIter shape retains its existing geometry. Threads remain fixed
at 512 because the kernel's internal `THREAD_NUM` mapping is compile-time 512.

## Four-rank graph sweep

The M64 attention-tail oracle used registered AIter buffers, rank-max timing,
100 mutations, 1000 graph replays and seven interleaved timing rounds. Every
geometry was exact and graph-stable.

| blocks | AR gross time |
|---:|---:|
| 16 | **30.146 us** |
| 24 | 30.649 us |
| 32 | 34.469 us |
| 48 | 37.024 us |
| 64 control | 43.301 us |
| 80 | 48.415 us |

Blocks=16 reduces the isolated collective by 30.4% versus 64. The fixed
owner/reduction order is unchanged; only the number of independent CTA ranges
and system-scope rendezvous participants changes.

## Service A-B-A plus independent default-B confirmation

Arm A explicitly set 64 blocks. Arm B explicitly set 16 blocks. The returned
A2 again set 64. A final independently started B2 omitted the variable and
therefore tested the new TP4 profile default. Each service ran three rounds;
B2 then ran three additional rounds to characterize slow/fast state variance.

Primary medians over six A and nine B observations:

| metric | A blocks64 | B blocks16 | change |
|---|---:|---:|---:|
| scheduler/model tok/s | 749.270 | 773.164 | **+3.19%** |
| aggregate HTTP tok/s | 674.096 | 692.276 | **+2.70%** |
| host speculative step | 72.850 ms | 70.594 ms | **-3.10%** |
| mean accepted length | 1.76344 | 1.76243 | -0.06% |
| common-resident HTTP tok/s | 648.787 | 640.224 | -1.32% |

The common-resident derived window moves opposite to scheduler, aggregate and
step time because heterogeneous requests enter/leave the exact common interval
at different positions. It is retained rather than hidden, but is not strong
enough to reject the direct model-step improvement. Blocks=16 is a roughly 3%
communication checkpoint, not a 5% claim.

All four independent services passed France exactly. Every one of the 480
measured coding requests returned 256 tokens with `finish=length`.

## Delivery

The TP4 BS32 profile defaults the environment variable to 16. A clean AIter
checkout must first apply, in order:

```text
scripts/rocm/patches/aiter_gfx90a_custom_ar_system_barrier.patch
scripts/rocm/patches/aiter_gfx90a_ar_512k_blocks.patch
```

The second patch was dry-run checked after applying the first to a clean AIter
HEAD archive. Rebuild `module_custom_all_reduce.so` with ROCm hipcc, not the
base-conda wrapper:

```bash
PATH=/opt/rocm/core-7.14/bin:/opt/rocm/bin:$PATH \
CPATH=/opt/rocm/core-7.14/include \
ROCM_PATH=/opt/rocm/core-7.14 MAX_JOBS=8 AITER_REBUILD=1 ...
```

An unpatched AIter safely ignores the environment variable, but receives no
performance benefit.

Artifacts:

- `/tmp/dsv4_ar512_blocks{16,24,32,48,64,80}.log`
- `/tmp/dsv4_dspark_ar512_{a64,b16,a2_64,b2_default,b2_default_more}.json`
- matching France JSON and service logs.

