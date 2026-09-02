# DSV4 gfx90a large-M BF16-CK correctness audit (2026-09-02)

The default-off `SGLANG_DSV4_GFX90A_BF16_CK_PREFILL` result at M<=36,864 is
not currently a small numerical drift.  A 32-request heterogeneous semantic
test produced severe repetition, while the production SDOT control was
coherent for all 32 requests.

Standalone isolation with the same expanded BF16 expert weights found:

- AIter sorting is populated and valid (`num_valid_ids`, sorted token IDs and
  expert IDs are nonzero).
- Generic CK with `ActivationType.Dsv4Silu` silently leaves stage1 all zero.
- Generic CK with ordinary `ActivationType.Silu` makes stage1 nonzero, proving
  the BF16 GEMM input/weight path is live, but stage2 still leaves the routed
  output all zero.
- `AITER_USE_NT=0` does not change either failure, so non-temporal loads are
  not the cause.
- Generated SiLU module contents show stage1 instances with `ActOP=1` and
  stage2 instances with `ActOP=0`.  The C++ wrapper remaps and forwards the
  activation value into stage2 even though stage2 has no activation.  Forcing
  the stage2 key to zero through the existing module still produced zero, so
  stage2 requires a smaller direct oracle before this selector can be enabled.

Acceptance policy: BF16 expansion/CK need not be bitwise equal to production
SDOT.  Small stable drift is acceptable after component error checks, France,
and heterogeneous C32 semantic validation.  An all-zero routed branch is not
acceptable and the selector must remain default-off until fixed.

## Deeper component isolation

Further probes separated two independent failures:

- Stage2 uses BF16 `AtomicAdd` for Top-6 reduction.  The CK gfx90a path reaches
  a BF16 atomic helper whose implementation is guarded to gfx942/gfx950/gfx12,
  so it performs no store on MI200.  FP32 atomic output restores writes, as
  expected from the MI200 ISA (F32 atomic add and packed-F16 atomic add exist;
  BF16 atomic add does not).  A correct implementation must use an FP32
  workspace plus BF16 cast, or a packed-BF16 CAS loop.
- After adding the missing DSV4 bounded-SwiGLU epilogue, a channel-identity
  probe showed only about 32 of 512 stage1 channels written on gfx90a.  This is
  not a harmless permutation.  The generic CK wave64 C-shuffle/writeback
  contract remains incorrect for this BF16 MoE instance.  Adding a BF16
  transpose specialization alone did not change the probe, because the active
  instance uses the same source/destination vector dimension.

With FP32 stage2 and the partial stage1 fix, M8192 measured roughly 17.5 ms for
the synthetic full routed stage, which is fast enough to remain promising, but
the numerical output is not usable.  Do not quote this as valid throughput.

For comparison, the correct production MFMA64 routed stage scaled negatively:

- M2304: 24.39 ms
- M4608: 54.31 ms raw; about 48.37 ms preshuffled at the standard grid
- M8192: 100.86 ms
- M16384: 208.53 ms

Large-M grid changes only moved M4608 by about one percent.  The missing C32
throughput therefore requires a correct high-occupancy kernel, not another
chunk-size or persistent-block-count sweep.
