# CK reuse review: sentinel test prepared, not run

The existing Python H8 selector only admits M128/192; its C++ wrapper and
common launcher also cap tokens at192. Do not remove those production guards
to test large prefill. A separate oracle would need its own checked entry.
The reusable implementation is the repo snapshot
`dsv4_unified_sparse_decode_ck.cuh`, not the absent external CK header initially
looked for. Its two-split workspace is M*2*8*514*4 bytes for H8.

Before any performance screen, inspect the sentinel contract. The core loads
out-of-range/negative KV as zero, while its score/probability masks use the
number of listed entries. The direct reference kernel explicitly skips such
slots. An analytic mixed-negative case with Q=0, valid V=1, sink=0 should
return1/2; counting a negative zero-V slot would instead give1/3.
All-negative/empty rows alone cannot reveal this, since their numerator is0.

`check_ck_sentinels.py` is prepared for physicalGPU4 after the current TP8
service ABBA stops. It tests valid, duplicate, empty, all-negative, and mixed
negative rows through the **existing** H8 module, recording output before
failing any exact comparison. No GPU result exists yet. Normal front-packed
production lists may never contain interior negative entries; this is not
evidence that the running AR/prefill service is affected.
