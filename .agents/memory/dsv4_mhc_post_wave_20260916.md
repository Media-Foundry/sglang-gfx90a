# Exact HIP wave-owned H256 MHC post

Original V4, TP8/EP1 native AR, original weights, 1M logical KV, C16x8K real
code tasks. Baseline is e4309a72de's corrected H16 + unique-slot CK + vec4
reducer configuration (~8.69k), not the older 5.57k/6.88k checkpoint.

## New work decomposition, distinct from failed two-pass fusion

The post-only reference processes one H256 tile with four wave64s. A CTA reads
x and all four residual channels, emits four output channels and their H256
RMS partials. This experiment reduces waves/CTAs without fusing pre-mix or
changing the coefficient dependencies. Output and `[M,64]` RMS scratch stay
the same; there is no new workspace, weight copy, precision change or KV cut.

First grouped-Triton screen merged H256 tiles into H512/H1024/H2048. AtM32767
the fastest changed4waves/H256 (~3.99ms) to4waves/H2048 (~2.11ms), but failed
both residual and RMS byte equality. These timings are rejected candidates,
not accepted model speeds. The same G1/4wave shape remained exact and flat.

## Arithmetic diagnosis and repair

`compiler_audit.py` launches the actual current reference and candidate before
capturing TTIR/TTGIR/LLVM/AMDGPU outputs; it does not inspect an arbitrary old
cache object. Reference scalar/packed code separately rounds c0*r0,c2*r2,c3*r3
and post*x, with one fused c1*r1 + round(c0*r0). Wider layouts generate more
packed FMAs and alter those rounding points. The RMS reduction tree also
changes with the thread layout. This is candidate-introduced drift, not proof
of a new cause for historical production-model drift.

The HIP repair explicitly uses that residual FMA/add sequence with contraction
disabled elsewhere. One wave owns an H256 tile, each lane holding the four
positions lane+64*q. For each quarter it reproduces the reference's DPP tree:
row_shr8 (with the original first square/add FMA), then4/2/1, row_bcast15/31.
Quarter totals combine as `(q0+q2)+(q1+q3)`, matching the old inter-wave tree.
A CTA has four waves / four H256 tiles, grid `[M,4]` instead of `[M,16]`.
No CTA barrier or LDS is needed. The tested object reports44VGPR,20SGPR,
wave64,0private scratch and0spills. Compiler/source/object hashes are archived.

## Component and integration gates

Initial HIP screen, repeated/scaled real layer0 samples, ten mutations/shape,
three graph ABBA cycles; residual and RMS byte-exact in all cases:

| M | Reference ms | HIP ms |
| --- | ---: | ---: |
|17|0.009795|0.010004|
|8192|1.003050|0.625560|
|32767|3.989905|2.538828|

Large-M latency improves about36%; small-M is not a speed win and remains
outside the selector. This is post-only component timing, not full MHC or E2E.

The integrated header then passed M1,17,128,8192,32767,32768,65536:100 mutations
each (including fresh random x/residual/post/comb), reverse-row permutation,
100 graph replays each except1000 atM128, stable pointers with input mutations
and poisoned outputs. An asymmetric integer comb fixture independently checks
the `[input_channel,output_channel]` orientation: outputs92/104/116/128 and
their exact H256 square partials. No empty-output checks count as agreement.

Default-off `SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE=1` is consumed only within the
existing `mix_pair_active()` strict native EXTEND scope and M8192..65536;
the outer fused4 wrapper checks actual shapes, contiguous layouts, dtypes and
gfx90a. V4.1, TP4, CP, TBO, draft, target verify and native decode do not enter.
22 CPU dispatch/scope tests pass. Global launcher defaults remain unchanged.

A separate fresh1M-pool TP8 service enabled `SGLANG_DSV4_DEBUG_POST_WAVE_CHECK=1`:
every selected post call also executes the current Triton reference. Across
four forwards:85 post boundaries x8ranks x4 =2720 complete residual AND RMS
byte comparisons, all exact. France passes before the large-prefill path.
The diagnostic's6429.3 input tok/s includes duplicate computation and checks;
it is NOT a performance sample or a regression. Formal timing disables it.

Artifacts: `.agents/experiments/dsv4_mhc_post_tiles_20260916/`.
The first compiler-audit filename `inspect.py` shadowed Python's stdlib inspect;
renaming it `compiler_audit.py` fixed import before any GPU measurement. First
HIP compilation also exposed a missing old-value argument to update_dpp;
the corrected six-argument intrinsic passed the gates above. Neither failed
attempt yielded valid performance data.

## Formal service result: +3.0976%

Every leg contains three warm timed waves; A1/B/A2 use three fresh processes,
B1/B2 are adjacent in one candidate process. Same131069-token input manifest,
zero prefix hits,one output token,32768 admission budget and actual1048576 pool.
The numeric/reference check flag is off in all timed arms. Every candidate
rank logs actual HIP post selection; the control has no such selection.

| Leg | Median input tok/s |
| --- | ---: |
|A1 reference post|8694.888536|
|B1 exact HIP post|8970.763958|
|B2 exact HIP post|8959.056579|
|A2 reference post|8696.226450|

Control center8695.557493; candidate8964.910269; **+3.09759%**.
Control A2/A1 drift+0.01539%. Runtime sources and input hashes match across
all arms. This is a post-only change on top of the previous vec4/H16 checkpoint,
not a combined comparison against the old MHC implementation from September15.

Each process additionally runs four C16x128-token code-continuation waves.
All16 requests are identical across all12 waves and both configurations:
192 checked responses /24576 output tokens, no empty arrays counted. France
passes in each process. Prompt echoes,request IDs,zero cache hits,128-token
lengths and token-to-text decoding are explicitly verified. This is bounded
same-workload reproducibility, not universal model/batch/logit invariance or a
proof of every generated code-review claim.

Retain the opt-in selector; no global default changes. `validated-launcher.sh`
contains the tested startup settings plus the five explicit outer lifecycle
environment values. It requires the packaged local CK/IPC manifests from the
previous rebuild experiment. Neither AR decode nor speculative paths changed.
No extra scratch is added; no model weights or quantization format changed.
All four services exited with no remaining owned children after measurement.

Next: refresh the same-rank critical-path profile before budgeting another
MHC/MoE optimization. Do not continue to count the old1.37-second post span as
entirely unoptimized, or assume the component's36% improvement applies to all
MHC. The rejected serial-column post-to-projection prototype remains rejected;
this result does not validate two-pass fusion by itself.
