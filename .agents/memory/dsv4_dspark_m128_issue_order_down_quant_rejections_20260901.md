# DSpark gamma-3 M128 issue-order and down-quant rejections (2026-09-01)

## M128 attention issue order

The TP4 BS32 profile inherited issue order 3 from the native M32 C4 tuning:
q_lora is issued before the core/index compressor branches.  Because M128 has
four times as many rows, mode 0 (launch both compressor branches first) was
screened with the same gamma-3, SBO-on, sparse graph tiers 1/32 and 32 distinct
code prompts.

Seven mode-0 resident rounds:

```text
948.518 / 924.126 / 834.185 / 902.654 / 885.694 / 840.347 / 991.727
median       902.654 tok/s
trimmed mean 900.268 tok/s
```

The adjacent mode-3 control median was about 901.625 tok/s.  Acceptance varied
strongly and the acceptance-normalized mode-0 median was about 371.87.  Every
round passed France first-nine exact + semantic Paris and 32x256
`finish=length`.  Mode 0 is neutral; keep issue order 3.

## Anchor-only intermediate quant

After grouped gate/up, production quantized the full `[128,6,512]` BF16
intermediate even though the model-side DSpark route leaves only rows `0::4`
valid.  A strict target-only HIP prototype quantized the 192 anchor assignment
rows and preserved the physical M128/A4/down layout.

Isolated quant:

```text
generic full-M128: 38.61 us
anchor-only HIP:     9.65 us
```

It matched valid-row INT8/scales exactly for 100 mutations, remained bitwise
stable across 1000 graph replays, and matched another 100 graph-input
mutations.

Service rounds were `859.350/917.079/841.317 tok/s` (median 859.350), below the
901.625 control despite all correctness gates passing.  A corrected complete
routed oracle explained why:

```text
generic gate->quant->down->reduce: 378.330 us
anchor quant full routed stage:     365.778 us
net stage saving:                    12.552 us (3.3%)
```

The full chain passed 100 activation/router-weight mutations bitwise.  The
12.55-us layer saving is only about 0.54 ms over 43 layers and did not survive
service variance/critical-path scheduling, so the prototype was removed.

## Oracle sentinel bug fixed

The first full-chain attempt produced a false mismatch because
`bench_dsv4_gfx90a_occupancy_bucket_oracle.make_metadata()` used
`buckets[expert]` without validating expert IDs.  A production `-1` sentinel
therefore became Python index `-1` and was silently routed to expert 255.
The helper now ignores IDs outside `[0,E)`, matching AIter's sorter contract.
This is diagnostic-only and does not change model execution.

## Decision

- Keep M128 issue order 3.
- Remove anchor-only down quant and its selector/carrier.
- Continue only with changes capable of removing a materially larger fraction
  of the roughly 365-us routed stage or the 280-us prepare region.
- Native AR was never eligible for either speculative prototype.

