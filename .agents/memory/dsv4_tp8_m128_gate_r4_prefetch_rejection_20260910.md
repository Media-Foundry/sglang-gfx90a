# TP8 strict-DSpark M128 gate R4 prefetch rejection (2026-09-10)

The accepted M128 routed path pre-issues both R2 gate/up packed rows before
decoding row zero.  An oracle extended only this schedule to R4.  Assignment
width remained A4, wave count remained eight, task ordering and all SDOT and
reduction arithmetic remained unchanged.  This is distinct from the rejected
A8, K-half/LDS-barrier, and expert-row-stripe experiments.

Physical GPU 4 only.  Both R4 variants passed 100 input/scale mutations,
matched the R2 BF16 intermediate, FP32 partial, and final BF16 output bitwise,
and were graph-replay stable.

Full gate -> quant -> unchanged down -> fixed reducer, synthetic M128:

| active experts / A4 scans | R2 B832 | R4 B416 | R4 B832 |
|---|---:|---:|---:|
| 244 / 293 | 773.86 us | 852.26 us | 824.80 us |
| 128 / 244 | 656.65 us | 720.25 us | 720.43 us |
| 32 / 203 | 571.60 us | 627.50 us | 625.12 us |

R4 regresses about 5.8--10.1%.  Issuing twice as many independent packed rows
before consumption does not compensate for the additional live register state
and/or reduced occupancy.  The temporary R4 source and benchmark wiring were
removed; production remains R2.  Do not repeat R4 without a design that reduces
live state rather than merely extending the prefetch distance.

Raw logs:

- `/tmp/dsv4_tp8_m128_gate_r4_b416.log`
- `/tmp/dsv4_tp8_m128_gate_r4_b832.log`

