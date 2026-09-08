# TP8 native M32 AR geometry: completed fresh-process ABBA

Experiment date: 2026-09-08 (Asia/Hong_Kong). Adapter checkpoint: 46918146e7.
The user requested stopping after A2. A2 is complete; no further experiment or default promotion follows this report.

## Scope and controls

Original safetensors, native AR, TP8/EP1/no-A2A, eight GCDs. The 1,048,576-token pool and context remain intact. Decode graph tiers: 1/2/4/8/16/24/32. NUMA interleave-all. Down-uniform and fixed down warmup remain enabled in every arm; attention issue order is 0. This changes only the registered native M32 all-reduce CTA count, not C1, prefill, or speculative dispatch.

Four independent service processes ran 4/16/16/4 blocks. Private launch snapshots and the final live process were compared: command and cwd identical; the only environment difference was SGLANG_DSV4_GFX90A_TP8_M32_AR_BLOCKS. Private snapshots are not committed. No explicit random-seed argument was introduced; generated runtime seeds may differ. The control uses the same compiled shim with 16 blocks, not the pre-existing library binary.

## Results (tok/s)

| Arm | PID | Blocks | C1 trimmed | C32 HTTP warm | C32 resident warm | C32 cross-wave exact requests |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 3632259 | 4 | 83.3151 | 1003.7413 | 1052.1442 | 9/32 |
| B1 | 3640693 | 16 | 83.5124 | 986.3847 | 1033.4958 | 6/32 |
| B2 | 3648726 | 16 | 83.9575 | 989.6716 | 1035.7339 | 9/32 |
| A2 | 3656477 | 4 | 83.4861 | 1001.4843 | 1049.4638 | 4/32 |

C1: four measured rounds of three fixed real code tasks; per task drop min/max, average, then task geometric mean. C32: six waves of 32 distinct code requests, 256 output tokens each; discard first wave and take the remaining median. Aggregate each pair of independent arms using geometric means.

| Metric | 16-block control | 4-block candidate | Change |
|---|---:|---:|---:|
| C1 | 83.7347 | 83.4005 | -0.3990% |
| C32 HTTP | 988.0268 | 1002.6121 | +1.4762% |
| C32 resident | 1034.6142 | 1050.8031 | +1.5647% |

This is a small repeatable C32 improvement in this ABBA, not a C1 improvement or a new 5% checkpoint. No confidence interval is claimed from two independent processes per arm.

## Correctness and capacity

- Each arm passed France 32/32; all 48 measured C1 completions match across arms for the full 256 tokens.
- All 768 C32 completions passed length/finish/native-AR checks and recomputed completion-ID hash integrity.
- C32 full outputs are NOT bitwise deterministic across all six waves, including both controls. Hash integrity and France correctness do not prove full semantic or numerical parity; do not claim this regression is resolved.
- A2 logs confirm actual blocks=4 selection on all eight ranks and full=1048576 on all ranks; runtime context_len=1048576. No extra persistent model/KV cache is introduced by the adapter. This is allocation preservation, not a filled-1M-context quality test.
- A2 reported available_gpu_mem=15.64 GB on TP0 after capture. Previous arms retained approximately 0.65 GB/GCD graph usage and 15.58–15.64 GB free.

## Audit and handoff

The adjacent JSON contains source artifact/state SHA256, workload hash, per-arm data, and aggregate metrics. Raw artifacts remain under /tmp using the recorded names; they are not durable cloud archives.

The summary tool now has --ar-geometry because the existing block candidate flag denotes down-uniform, which is true in all four arms. Geometry mode validates 4/16/16/4, four distinct PIDs, constant down-uniform/attention order, and fixed-process completion before normalizing candidate labels. Eight CPU regression tests passed.

Adapter stays default-off (AR_BLOCKS=0); no global configuration promotion. Final A2 service PID 3656477 remains running on 127.0.0.1:30011 with blocks=4. Work stops after recording and committing this result, per user request.
