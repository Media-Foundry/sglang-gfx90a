# Request-aligned projection experiments

Same real layer0 fixture and unchanged diagnostic service2878897. PhysicalGPU4
only with idle service; no external GPU PID at precheck. No production changes.

Padding individual46-token request matrices to48/64/96/128 rows then flattening
to one GEMM does NOT repair row invariance. Exact valid rows were612/612/638/612
of736 (unmodified46 stride617/736), maxdelta0.0009765625 throughout. Do not
add per-request padding to attention/KV based on the earlier tile-offset guess.

Instead torch.bmm over[16,46,4096] with weight transpose expanded as a zero-stride
view[16,4096,1536] yields736/736 identical rows; it does not allocate16 weight
copies or serialize16Python GEMM calls.100/100 repeated-input mutations pass.
Maximum discrepancy against first46 FP64-reference rows roundedBF16 is
0.000244140625. Five graphABBA cycles: current flattenedmm89.416us versus
request-bmm154.256us. Faster than initial211us fixed-order Triton but slower
than its tuned113us and slower than production. This is a diagnostic repair
candidate, not an accepted E2E fix or speedup.

It demonstrates that preserving a request-local matrix layout can eliminate
the observed slot dependence without modifying attention semantics or weight
precision, but general variable-length requests and other projections still
need integration and validation. Equal-length16x46 is explicitly hardcoded
only in the standalone --request-bmm option. No deployment flag added.
Raw:/tmp/dsv4_slot_request_bmm_20260908.log.
