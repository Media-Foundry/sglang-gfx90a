# DSV4 M128 hot-expert LDS staging rejection (2026-08-31)

## Decision

Reject the oracle-only hot-expert LDS-staged gate/up and down decomposition.
It is bitwise exact, including HIP graph replay, but substantially slower than
the production A4/R2/W8 routed-FP4 kernels.  Do not add a production selector
for this design.

No production source was changed.  The experiment consists only of:

- `python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_hot_expert_lds_oracle.cuh`
- `scripts/rocm/bench_dsv4_dspark_m128_hot_expert_lds.py`

## Resource isolation

The pre-run `amd-smi process` check reported no processes on any GCD.  The
formal run used physical GCD 4 only via `HIP_VISIBLE_DEVICES=4`.  The process
exited after the result was written; no service or benchmark process was left
resident.

## Workload and route

The oracle uses the middle complete M128 record from the real gamma-three
recorder:

- recorder: `/tmp/dsv4_gamma3_recorder2/expert_distribution_recorder_1788146391.4818711_0.pt`
- record index: 38
- forward pass: 22
- layer: 20
- routed assignments: 128 x 6 = 768
- active experts: 125
- maximum occupancy: 117
- production A4 scans: 250

The static hot threshold is `run_len > 8`:

- hot experts: 22
- hot assignments: 424 (55.2%)
- hot A4 scans: 116
- cold assignments: 344
- cold A4 scans: 134

Cold experts use the unchanged production A4/R2/W8 kernels.  For hot gate/up,
one wave owns an `(expert, output-row)` pair, decodes the packed gate and up
row into wave-private LDS once, then traverses all of that expert's A4 chunks.
For hot down, one 16-lane subgroup applies the same scheme to an
`(expert, output-row)` pair.  Quantization, FP32 fixed-slot partials, and the
fixed-order final reduction are unchanged.

## Correctness

- Initial output: exact at every boundary.
- 100 eager input/weight mutations: bitwise exact.
- 1,000 HIP graph mutation replays: bitwise exact.
- Checked tensors: BF16 gate/up intermediate, INT8 activation, FP32 scales,
  FP32 down partial, and BF16 reduced output.

This rules out stale LDS, graph-capture state, assignment-slot corruption, and
reduction-order changes for the tested shape.

## Symmetric ABBA result

Seven rounds used the symmetric order `A, G, GD, GD, G, A`, where:

- `A`: full production gate + production down.
- `G`: cold production gate + hot LDS gate + full production down.
- `GD`: cold production kernels + hot LDS gate and hot LDS down.

Trimmed means in microseconds:

| Stage | A | G | GD |
|---|---:|---:|---:|
| gate/up | 713.057 | 1,145.790 | 1,145.723 |
| activation quant | 19.169 | 19.656 | 19.578 |
| down | 524.236 | 523.897 | 812.684 |
| reduction | 13.573 | 13.288 | 13.353 |
| full routed stage | 1,267.785 | 1,709.317 | 2,002.907 |

Derived changes versus production:

- mixed gate/up: **37.77% slower** (`713.057 -> 1,145.790 us`)
- gate-only full routed stage: **25.83% slower**
- gate+down full routed stage: **36.70% slower**
- required continuation threshold: at least 15% faster
- outcome: fail by a wide margin

Raw JSON: `/tmp/dsv4_m128_hot_lds_gpu4.json`.

## Why the apparently useful reuse loses

The experiment does remove explicit packed-weight decode from every hot A4
chunk, but it exchanges independent row-chunk tasks for long serial per-row
work:

1. Production exposes every `(expert, A4 chunk, output row)` as independent
   work.  The hot kernel makes a wave/subgroup traverse all chunks for its row.
   The expert with occupancy 117 therefore creates a 30-chunk serial loop and
   a long-tail wave, even though most experts are much shorter.
2. Consecutive production A4 chunks already read identical packed weight rows.
   Natural L1/L2 reuse makes the raw HBM-saving assumption much smaller than a
   count of source-level loads suggests.
3. Explicit LDS adds decode stores, LDS reloads, address work, and a larger
   per-CTA shared-memory footprint.  Gate/up is especially expensive because
   both 4096-wide rows and both scale rows are staged.
4. Down has more subgroups per CTA, but serial traversal still reduces the
   enormous row-chunk task pool and worsens the hot-expert tail.

The result supports the earlier lesson from M64 expert-row persistence: for
this routing distribution, preserving independent A4 row-chunk scheduling is
more valuable than explicitly retaining decoded weights across chunks.

## Follow-up constraint

Do not retry this design by merely tuning the hot threshold or grid size.  A
new attempt must retain production-scale row-chunk parallelism while sharing a
decoded row without CTA-wide synchronization, or first demonstrate through
hardware counters that production A4 actually misses the repeated weight row
in cache.  Otherwise continue with distribution-level tactic selection or a
different work decomposition.
