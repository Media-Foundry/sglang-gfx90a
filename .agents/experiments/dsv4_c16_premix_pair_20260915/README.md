# Pending independent-column pre-mix screen

NOT GPU-tested. Do not run until the mixed-prefix service ABBA releases all
GPUs. No production selector or measured source changes.

Current accepted pre-mix8 uses one output column and eight rows per CTA.
This candidate keeps eight rows and computes two independent output columns,
attempting to reuse each input load. Each reduction remains shape[1,1024],
with the same16 K chunks and FP32 accumulation/RMS expression. This differs
from changing BLOCK_N to2, which historically changed the reduction tree at
M128 (`dsv4_tp8_m128_mhc_fusion_exactness_barrier_20260910.md`). Nevertheless
compiler changes can still alter results or register pressure: assume neither
exactness nor speed before the oracle runs.

Run physicalGPU4 with DS conda Python and repository PYTHONPATH:

```
HIP_VISIBLE_DEVICES=4 SGLANG_USE_AITER=1 python screen.py --output screen.json
```

Ownership guard executes before importing Torch. Compares current production
pre-mix8, repeated/scaled captured layer0 residual/Fn and original K1024/RMS
order, ragged M, row permutation,1000 graph replays plus changed-input replay.
Save failed comparisons before asserting. Full-component eager ABBA includes
RMS consumption; register/spill/LDS and HSACO hash are retained. This is not a
new service benchmark. Continue only if exact and consistently about10% faster
on large relevant shapes; otherwise retain production pre-mix8. Stronger random
Fn/input mutations and real-layer captures remain required before integration.
