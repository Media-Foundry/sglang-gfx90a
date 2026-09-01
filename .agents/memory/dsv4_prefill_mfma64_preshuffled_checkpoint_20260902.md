# DSV4 MFMA64 direct access to AIter-preshuffled weights (2026-09-02)

## Goal

Use one AIter A16W4-preshuffled routed-expert weight copy for both the custom
gfx90a direct prefill path and AIter's high-occupancy path.  This is the memory-
viable prerequisite for a unified C1/direct plus C32/AIter hybrid; duplicating
the routed weights is not viable at the target KV capacity.

## Implementation

The existing scalar address probe had already validated AIter's real
`shuffle_weight_a16w4` mapping.  The MFMA32/64 gate and down templates now have
a default-off `kPreshuffled` mode which uses those byte offsets while retaining
the established shuffled E8M0 scale mapping and exact accumulation order.
Production selectors and the loader are unchanged.

The standalone oracle compares:

```text
raw packed weight + shuffled scale
vs
AIter-preshuffled packed weight + shuffled scale
```

for gate/up, BF16 bounded-SwiGLU output, group32 INT8 values/scales, down, fixed
FP32 reduction, and final BF16 output.

## Correctness and performance

Physical GPU 4, gfx90a, M2304/top-6/E256/H4096/I1024/N4096, MFMA64:

```text
CORRECTNESS mutations=100 all_exact=True
raw trimmed:           24506.925 us
preshuffled trimmed:  22814.327 us
gain:                       6.91%
```

Every mutation used unique per-token top-k experts, matching the router
contract.  An earlier synthetic generator used independent `randint` experts
and therefore produced duplicate experts within a token; AIter's sorter does
not promise that invalid input contract and it caused misleading nonfinite
outputs at large M.  With unique top-k, M32, M512, M1024, and M2304 checks are
exact.  M1024 also measured about 3.9% faster in a short ABBA.

## Remaining gate

Do not enable preshuffling in the loader yet.  The accepted decode path still
uses raw-layout grouped/GEMV kernels.  Next, add the same default-off addressing
mode to basic grouped gate/down, validate M1/4/8/16/32/64 exactness and ABBA,
then migrate the LDS/DPP specializations.  Only after France, teacher-forced,
C1 prefill, real diverse C32 prefill, and decode ABBA pass may one shared
preshuffled layout become a service profile.
