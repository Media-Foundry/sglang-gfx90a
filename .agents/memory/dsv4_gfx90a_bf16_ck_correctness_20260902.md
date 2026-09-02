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

