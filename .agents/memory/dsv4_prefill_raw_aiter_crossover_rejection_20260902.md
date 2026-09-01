# DSV4 raw-weight AIter prefill crossover rejection (2026-09-02)

AIter exposes and generates separate FP4 MoE module families for
`preshuffle_on` and `preshuffle_off`, so an experiment kept the accepted raw
weights for small-M direct MFMA64 and routed only M>=8192 to AIter with
`preshuffle=False`. This would have enabled a large-M grouped kernel without a
second weight copy.

On physical GCDs 4--7, chunk budget 16384 and the fixed 32-request diverse code
workload, the first C32 round reached only about 1498 input tok/s. More
importantly, all completion witnesses diverged broadly from both the current
direct path and the separately loaded preshuffled AIter oracle. This is not an
acceptable reduction-order difference: the DSV4 raw weight/scale layout does
not satisfy the generated preshuffle-off module's full ABI.

The crossover selector and environment variable were removed. Keep the
existing fail-loud guard; a future raw CK grouped path must first define and
validate the exact weight-scale/nibble layout rather than assuming that
`preshuffle=False` alone is sufficient.

A second service forced `SGLANG_USE_AITER_MOE_GU_ITLV=0` so raw separated
gate/up weights did not enter the default interleaved activation contract. It
still produced broadly different completion witnesses and only about 1922
input tok/s for C32. Thus gate mode is not the sole missing ABI transform.
