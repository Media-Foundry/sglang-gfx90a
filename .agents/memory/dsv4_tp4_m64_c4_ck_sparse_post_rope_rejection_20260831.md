# DSV4 TP4 M64 C4 CK sparse + HIP post-RoPE rejection (2026-08-31)

## Hypothesis

The accepted CK-style MFMA sparse-decode kernel only serves C128 layers because
C4 needs inverse RoPE on the final 64 output dimensions.  An opt-in candidate
reused the same split-K=2 CK sparse core for C4, then launched a graph-safe HIP
kernel that read the BF16 attention result, applied inverse RoPE in FP32, and
rounded the final pair to BF16 exactly like the Triton reference's materialize
then rotate order.

## Scope and correctness

- Original DeepSeek-V4-Flash weights, native AR, TP4/EP1/no-A2A.
- Physical `HIP_VISIBLE_DEVICES=4,5,6,7`.
- Graph tiers 1/64, 65,536-token pool.
- 64 distinct concrete prompts, 256 generated tokens each.

The v4 JIT module compiled and all graph tiers captured.  The France completion
was exact and matched the baseline SHA256:

```text
5aada90c1ebf8e823aa8c419390c0d2928ad5206a1ec75aa79eb54a82d70ad66
```

All benchmark requests returned 256 tokens with `finish=length`.

## Result

Three-round medians:

| arm | resident tok/s | scheduler tok/s | host step |
|---|---:|---:|---:|
| accepted C128-only CK | 1038.31 | 1046.97 | 61.129 ms |
| C4 CK + separate HIP inverse RoPE | 1029.80 | 1038.69 | 61.616 ms |

The candidate regressed resident and scheduler throughput by about 0.8%.  The
extra post-RoPE launch and output read/write cost more than the CK sparse-core
gain on the actual C4 lengths, and unlike the C128 branch it did not improve
the end-to-end critical path.

Artifacts:

```text
/tmp/dsv4_native_m64_rebaseline_4567.json
/tmp/dsv4_native_m64_ckc4_b1.json
/tmp/dsv4_native_m64_ckc4_france.json
```

## Decision

Reject and fully remove this production candidate.  Keep the accepted
C128-only CK selector unchanged.  A future C4 attempt must fuse inverse RoPE
inside the CK reduce kernel (with no third launch and no extra output pass) and
first demonstrate a component win on real C4 lengths; do not retry a separate
post-processing kernel.
