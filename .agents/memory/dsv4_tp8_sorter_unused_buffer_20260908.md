# TP8 unused sorter buffer oracle — 2026-09-08

Base15c508e131. Production PID3277900 remained unchanged, native TP8/EP1 and KV pool1048576. AMD-SMI audit found no foreign GPU PIDs before isolated GPU4 probe.

Current custom grouped runner discards moe_sorting's _moe_buf; test passing model_dim0 instead of4096, retaining AIter sort, block_size4, metadata capacities and arithmetic. No production patch.

Existing AIter extension module_moe_sorting.so accepted a zero-width buffer. Tested M1 andM32 with100 changing unique-top6 ID/weight inputs each: valid sorted IDs, weights, expert blocks and both metadata words bitwise equal. Unused capacity tails are not compared.1000 graph replays per shape stable. M1 is API coverage, not proof the C1 production runner uses grouped sort.

Five ABBA cycles,100 replays/sample,trim extrema:
M1 baseline12.024190us versus empty11.981187us.
M32 baseline10.168758us versus empty10.208759us.
No material speed benefit, do not integrate. Nominal unused output storage is8KiB/256KiB per invocation; not a measured reduction in graph pool or physical peak memory. Sorter correctness alone does not prove complete routed-stage or service parity.

Script: scripts/rocm/bench_dsv4_sorter_unused_buffer.py.
Raw samples in adjacent JSON. Original weight precision, service configuration and1M pool unchanged.
