# Concurrent fixed-prefix numerical audit

Unchanged1M-pool server2857900. Client script check_dsv4_concurrent_fixed_prefix.py
uses six established fixed prefixes repeated across32 barrier-start clients,
fresh cache salts, one output token, full supplied-continuation logprobs/top20.
Three waves; no free-running autoregressive divergence in the comparison.
This is a prefill batching test, NOT a fixed-M32 cached-decode oracle or speed
measurement. Actual scheduler batching can differ between waves.

All96 next-token IDs match sequential reference. None of the96 input-logprob
or output-top20 arrays is bitwise identical to reference. Across waves all32
next-token IDs remain exact; zero of32 full logprob arrays are identical.
For matching token IDs, maximum input logprob absolute difference1.921249;
common output top20-token logprob difference up to3.729107. These are logprob
differences, not logits/hidden-state error. Do not label them harmless roundoff
just because next-token IDs match. No new production changes caused this test.

This narrows a correctness limitation: concurrency-dependent prefill numerics
already exist before cached decoding begins. It does not establish a race;
batch shape can change GEMM/CK backend and reduction order. Follow up by
holding the actual GPU batch shape fixed, comparing current BF16-CK selector
with a reference path, and locating first divergent layer/selected experts.
Do not infer that disabling recently accepted native-M32 decode switches will
fix an observed prefill-only difference. Keep user-withdrawn paths withdrawn.

Raw full response artifact:
/tmp/dsv4_tp8_1m_concurrent_fixed_prefix_20260908.json
Client session96195 completed successfully. Existing C1 exact checks and
France semantics remain true but cannot support full C32 numerical equivalence.
