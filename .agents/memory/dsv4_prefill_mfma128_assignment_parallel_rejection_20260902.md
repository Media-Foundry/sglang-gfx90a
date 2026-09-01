# DSV4 prefill assignment-parallel MFMA128 rejection (2026-09-02)

To address the C32 large-M routed-MoE weight-scan limit without repeating the
failed 48-KiB MFMA96 accumulator design, an oracle mapped one A128 expert block
to an eight-wave CTA. Each wave owned 16 assignments; all waves reused a single
raw FP4 gate/up N16 weight tile staged in LDS for each K group. This kept
per-wave accumulators bounded and reduced logical expert scans from roughly six
to three at M16384.

Physical GPU 4, balanced E256/Top-6/H4096/I512, ABBA medians:

```text
production A64 split-K gate: 52.924 ms
A128 assignment-parallel:    64.611 ms
delta:                       +22.08% (regression)
```

After fixing an initial missing wave broadcast, the numerical difference was
consistent with the changed FP32 reduction order (`max_abs=0.5`,
`mean_abs=9.10e-6`) rather than an address/layout failure. The performance
result is decisive: staging each group needs a CTA barrier before consumption
and another before overwrite, so 256 barriers over K4096 cost more than the
saved global weight reads.

All oracle code was removed. Do not retry A128/256 by inserting per-K-group CTA
barriers. A future large-M design needs an asynchronous/double-buffered producer
or a work decomposition that avoids both large resident accumulators and a
barrier pair per group.
