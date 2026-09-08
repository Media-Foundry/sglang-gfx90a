# TP8 gate LUT8 replication screen — 2026-09-08

Baseline source: 3c815d5c64. Production service PID 3277900 remained TP8/EP1/native with 1,048,576 KV tokens. AMD-SMI process audit found no foreign GPU PID. Only the standalone probe used physical GPU4; no production selector changed.

## Motivation and resource audit
The tested LUT32 binary has VGPR95, SGPR52, LDS32768 bytes, private segment0 and zero spills. Baseline previously audited VGPR95/LDS1024. This motivates testing a smaller replication factor, but does not prove bank conflicts or occupancy caused the slowdown.

## Test
Existing bench_dsv4_tp8_gate_prefetch_screen.py --lut-replicas 8.
Five ABBA cycles, 100 graph replays per timed sample; trim one minimum and maximum from ten samples per arm.
The measured chain includes gate, intermediate quantization, down partial and fixed reduction; excludes sorter and initial input quantization.
Synthetic routing distributions, not real end-to-end requests.

| Active experts / scans | Baseline us | LUT8 us |
|---|---:|---:|
| 133 / 133 | 272.446 | 275.650 |
| 104 / 105 | 242.262 | 245.486 |
| 32 / 61 | 182.355 | 185.140 |

All 300 mutation cases had exact intermediate/final output and FP32 partials, max absolute error0. Candidate replay stability passed 1000 replays per distribution.
LUT8 is slower in all three cases (about 1.18–1.53% latency). Do not integrate or claim E2E gain. Combined with LUT32 rejection, stop replica tuning absent new evidence; factor16 remains unmeasured.
No production numerical or VRAM change; no new E2E correctness/performance claim.

