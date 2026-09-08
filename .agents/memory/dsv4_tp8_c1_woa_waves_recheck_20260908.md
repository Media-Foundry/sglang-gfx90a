# TP8 C1 wo_a wave-count recheck: no opportunity

Baseline1475379874, service3391501 unchanged, 1M KV pool. PhysicalGPU4 only;
AMD-SMI PID audit had no foreign processes. No production headers or selectors
changed. Standalone transient weight bank64MiB released when process exited.

Hold M1/G1/N1024/K4096, rows1, unroll2 and exact FMA/shuffle order fixed; compare
2/4/8 waves. Five symmetric cycles4/2/8/8/2/4, ten samples/variant, trim extrema.
Each timed graph has64 operations. Hot repeats one8MiB weight; rotating8 cycles
64MiB total weights. Synthetic values, not a full model or cold-HBM measurement.

| Waves | Hot us | Rotating8 us |
|---|---:|---:|
| 4 current | 7.11426 | 9.40795 |
| 2 | 8.03112 | 10.53450 |
| 8 | 7.31863 | 9.98325 |

Both candidates:100/100 mutation sets (eight weight/input pairs each) raw BF16
bit-pattern exact, finite;1000 graph replays stable and allocated memory unchanged.
No service test warranted and no new E2E speedup.

## Important audit correction

The initial search used only experimental_switches.md and missed the separate
`dsv4_tp8_c1_woa_screen_20260908.md`. Full-memory search found this same geometry
screen already completed with43 rotating weights and the same negative ranking.
This run is redundant confirmation, NOT a novel optimization. The old direct-X
activation-staging variant is also already documented/tested; do not reimplement
it as a new idea. Preserve the current standalone script/raw timing for audit,
but close wave-count/rows/direct-X scans unless a materially different mechanism
or production shape is demonstrated first. Search all memory files before the
next hypothesis, not only the large consolidated switch table.

Production C1/C32 speeds remain unchanged; the last full baseline repeat was
~83.63 C1,985.13 C32 HTTP,1031.29 C32 resident tok/s. These are historical
same-run baseline values, not rates measured during this component test.
