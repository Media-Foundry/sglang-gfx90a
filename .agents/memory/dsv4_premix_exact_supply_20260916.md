# Exact cooperative pre-mix supply: small gain, wider groups rejected

2026-09-16, base21f9787ff5. Experiment only; no production selector changed.
Current exact-reference prefill remains approximately8.96k; separate MFMA
numerical alternative is measured9.43k and default-off.

## Actual arithmetic, not just the Python expression

Pinned the returned current `premix8_pair` Triton object atM32767. The cached
debug locations name the identical historical candidate source; the object was
obtained by invoking the current production callable, not selecting an arbitrary
cache file. Layout: four contiguous K values/lane, repeated at offsets256/512/768.

Each K1024 chunk/lane starts with rounded product1, then FMA product0, then
products2..15. DPP tree: row_shr8/4/2/1, row_bcast15 mask0xa, row_bcast31;
take lane63, then sum16chunks in order. RMS uses the same64-lane tree and
FMA(sum,1/16384,1e-6), hardware rsq, final multiplication.

Critical exception found in the actual generated ISA: chunks13/14 compute their
last two products with packed multiply plus separate adds; chunk15 does this for
the last four. All16row/column dot streams show FMA counts15x13,13,13,11 and
packed-multiply counts0x13,1,1,2. `analyze.py` audits this from the pinned object.

The initial all-FMA HIP guess was not exact (about1e-6 max abs in small random
checks). Explicitly reproducing these tail exceptions removes the discrepancy
in all measured inputs. Reversing the initial product/FMA order remains a negative
control. This is **current compiler-specific reference reproduction**, not a
portable mathematical identity or a newly discovered production bug.

An initial capture script named inspect.py shadowed the Python standard library;
renamed to capture_reference.py before the actual capture. No environment failure.

## Work decomposition

Each wave retains production8rows and2independent output columns.2/4/6/12waves
share one BF16 activation tile[8,1024] in16KiB LDS; each keeps its own FP32 Fn
vectors and exact arithmetic. No global scratch.16bounded K chunks, two CTA
barriers/chunk; no global task queue, cross-CTA waiting, or changed model precision.

## Results: full pre-mix and post+pre-mix

Three graph ABBA cycles,five calls/timing sample; sampled real layer0 inputs
repeated/scaled for occupancy, not independent live full-M inputs.10input/Fn
mutations per shape/candidate, random row permutation and100graph replays pass
byte equality against production. M8192/32767/32768/65536 all pass, including
Triton's different divisibility specializations.

M32767, milliseconds:

| Candidate | Pre-mix | Post + pre-mix |
|---|---:|---:|
| current paired reference |~4.96|~7.37|
| HIP direct supply |24.53|26.94|
| shared2waves |8.49|10.90|
| shared4waves |4.70|7.11|
| shared6waves |5.90|8.33|
| shared12waves |5.62|8.05|

Shared4 full-boundary improvement is about3.5% atM32767,3.2% atM32768,
6.5% atM8192 and1.8% atM65536. This is not a service gain. Shared4 reports
82VGPR,20SGPR,16KiB LDS,zero scratch/spills in the inspected device object.
Increasing group width is not beneficial; no hardware-counter root cause claimed.

## Decision and next priority

Retain the exact arithmetic oracle and shared4 prototype; no service integration
yet because the net boundary saving is onlyabout0.26ms atM32767. A new reviewer
suggestion—owner computes only24pre-mix values and gathers those, while leaving
all residual state replicated—offers a larger potential work reduction and is
the next bounded investigation. First prove rank input identity and local-slice
equivalence; never synchronize away pre-existing rank differences.

Artifacts `.agents/experiments/dsv4_premix_exact_supply_20260916/` include failed
initial/tail checks, source snapshots, narrow/wide timing JSON and reference ISA/IR
archive. GPU work only HIP_VISIBLE_DEVICES=5 (PCI B3:00.0); no service overlapped.
