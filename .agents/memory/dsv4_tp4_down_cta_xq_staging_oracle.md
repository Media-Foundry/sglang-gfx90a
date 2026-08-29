# TP4 M32 grouped-down CTA activation staging oracle (2026-08-30)

## Candidate

Oracle-only A4/R2/W8/D832 kernel based on the exact row-prefetch down path.
For each CTA-uniform task iteration, all 32 subgroup tasks address consecutive
rows of the same expert/A4 assignment block.  The CTA cooperatively stages the
four K512 INT8 activation rows and their group-32 scales in LDS, then preserves
the existing packed-FP4 weight loads, pair-LUT decode, SDOT order, subgroup16
reduction, FP32 partial layout and fixed final reduction.

The mapping is guarded by compile-time divisibility checks.  All threads,
including invalid/tail tasks, execute both the publish and overwrite-prevention
barriers.  No production selector or wrapper was changed.

## Static resources

The emitted gfx90a descriptor reports:

- 45 VGPR and 54 SGPR;
- 3376 bytes LDS;
- zero private segment, VGPR spill or SGPR spill;
- wave64.

Thus the candidate passes the predeclared static resource gate.

## Correctness and ABBA

The real BS32 diverse recorder route (pass 37, layer 34) was used.  One hundred
mutations were exact for intermediate BF16, down FP32 partial and final BF16.
Seven rounds used `A/B/B/A`, with A equal to the current row-prefetch down path.

| stage | A trimmed mean | staging B trimmed mean | change |
|---|---:|---:|---:|
| down | 168.872 us | 194.938 us | +15.44% |
| full routed | 434.880 us | 462.421 us | +6.33% |

The candidate is correct and resource-light but slower.  The extra cooperative
global-to-LDS writes, two barriers per CTA iteration and repeated LDS reads cost
more than the activation global loads they replace; those original loads are
already cache-friendly across adjacent row tasks.

Decision: reject CTA-wide xq/x_scale LDS staging for TP4 grouped down.  Do not
wire it into production or repeat it with gate/up, whose staged tile and barrier
cost are substantially larger.
