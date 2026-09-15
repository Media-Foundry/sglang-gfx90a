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
