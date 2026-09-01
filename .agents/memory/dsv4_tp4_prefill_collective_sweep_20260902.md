# DSV4 TP4 large-prefill collective sweep

Date: 2026-09-02

The exact graph-replay oracle compared AIter custom peer-read collectives with
RCCL on physical GCDs 4,5,6,7 for the TP4 hidden tensor `[M,4096]` BF16.

| M | Payload | AIter RS | RCCL RS | AIter AG | RCCL AG |
|---:|---:|---:|---:|---:|---:|
| 2048 | 16 MiB | 311.9 us | 421.7 us | 497.8 us | 412.4 us |
| 2304 | 18 MiB | 348.9 us | 461.1 us | 459.0 us | 423.0 us |
| 2560 | 20 MiB | 384.1 us | 464.6 us | 526.3 us | 462.6 us |

AIter remains clearly better for reduce-scatter, while RCCL wins all-gather.
The best mixed AIter-RS plus RCCL-AG pair improves the combined component by
only about 4.5%, which projects to less than 1% TTFT. No production backend
switch was made. Revisit only if a fused boundary removes a launch/event or if
the collective becomes rank-max dominant after MoE optimization.

The retained oracle is `scripts/rocm/bench_dsv4_tp4_prefill_collectives.py`.
