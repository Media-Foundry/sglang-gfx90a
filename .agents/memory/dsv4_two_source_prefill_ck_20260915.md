# Two-source prefill CK: independent ABI, inherited numerical issue reproduced

Production remains the accepted7367.67 input tok/s configuration. New prototype
is experiment-only and does not widen production decode selectors. An independent
CK derivative and checked M1..65536/H8/D512 entry read prefix and extend banks
directly; a small GPU index encoder sums existing indptrs without D2H or KV copy.

20 analytic cases and exact index mapping passed. Random mutation34 failed the
predeclared FP32 tolerance; the failure is saved. Graph testing was not reached.
New two-bank CK output equals original single-bank CK byte-for-byte after a
diagnostic-only concatenation of the small synthetic tables. Production Triton
is much closer to FP32 at the failed element. Thus this is not evidence of a
bank-address bug; inherited BF16 probability rounding and changed region/tile
grouping are the next numerical suspects. This does not identify the old service
drift. No performance claim or production integration yet.

Evidence: `.agents/experiments/dsv4_two_source_prefill_ck_20260915/`.
Failure fixture SHA256:
`4f279a90c08eb56c1a67cd4deee4182857a3fdfcaf5acd17b13b8a0662333fbe`.
Next try existing hi+lo probability decomposition at unchanged tolerance, then
full-chain timings. Preserve failed evidence and production source/guards.

## Follow-up complete: hi+lo passes; direct CK reuse rejected for speed

Experimental-only hi+lo probabilities pass the same20 analytic cases,100 random
Q/two-KV/sink/index/ragged-boundary mutations at unchanged atol.003/rtol.02,
and1000 splits2 graph replays plus changed-input replay. Maximum absolute error
.0077226162 is under the combined absolute+relative criterion, not a standalone
.003 absolute bound. The frozen failed element now equals Triton at-.0654296875
(FP32-.0654088557). This repairs the prototype's inherited rounding issue, not
the production model's unresolved drift. Initial failure remains in1785bad8c8.

PhysicalGPU4 consumer ABBA, including encoder+CK+reducer, contiguous Q:
M8192 C1281.990->4.084/4.622ms (split1/2), C4 reused5.933->12.700/13.149ms,
C4 varied5.943->12.687/13.139ms.
M32768 C1288.252->16.139/18.339ms, C4 reused24.166->50.323/52.292ms,
C4 varied24.161->50.314/52.278ms. These are synthetic causal8K index patterns,
not actual service activations. Sampled reference checks/all-output finiteness
passed. Allocated split2 workspace1.078GB atM32768, C4 combined indices75.36MB;
split1 could halve scratch but still loses badly. No strided-Q or service test.

Reject this direct decode-core transplant; preserve the numerical contract
fixture for future purpose-built kernels. Do not repeat a generic CK-vs-Triton
claim based on decode M128 numbers. Production remains7367.67 input tok/s;
no original production CK header, attention selector or launcher changed.
Both benchmark processes exited0 and all GPUs were confirmed free afterward.
