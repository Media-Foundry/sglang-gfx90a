# TP8 C16 FP32 MFMA pre-mix service trial

Default-off `SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA=1` changes the FP32 reduction order.
It does not quantize Fn, change checkpoint weights, or skip target computation.
Do not label this bit-exact to the accepted paired pre-mix.

Order: `integrated.py` on HIP_VISIBLE_DEVICES=5 (PCI B3:00.0); existing CPU scope
tests plus `test_dsv4_prefill_mix_mfma.py`; then `service.py --arm check`, `A1`, `B`,
`A2` sequentially. All service arms own TP8 exclusively and stop only their owned
process trees. Use DS Python and /opt/rocm as ROCM_HOME/ROCM_PATH and PATH prefix.
Result directories refuse overwrite. Never overlap GPU microbenchmarks and service.

`check` runs live pre-mix versus legacy at every eligible boundary. Timing arms
disable checks, use three waves/leg, and four fresh-cache 128-token quality waves.
They additionally teacher-force 64 tokens from the prior accepted reference on
the same 16 code prompts. The response's first logprob/top-k position is a null
placeholder: `analyze.py` checks and excludes it, evaluating **1008**, not1024,
positions. These are fixed-prefix selected-token logprobs and top5, not full-vocab
logits. `archive.py` preserves raw evidence after analysis.

The service uses the accepted exact HIP post, unique fixed-slot CK, corrected H16,
32K chunk, original V4 and 1M logical KV in both arms. Only pre-mix differs.
Performance, within-arm repeatability, cross-arm differences and quality limitations
must be reported separately. No universal batch-invariance claim.
