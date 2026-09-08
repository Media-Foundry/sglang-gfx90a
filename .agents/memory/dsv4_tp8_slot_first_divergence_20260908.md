# Slot-dependent numerics: first divergence isolated

User asked to repair. Original1M baseline2857900 stopped after preserving
command/env. Existing stage/attention debug hooks enabled only layer0/rank0;
no production source modifications, original dirty model edits preserved.

First diagnostic2871592 used rows4096 based on padded log and missed stage
hooks. The unconditional legacy KV debug emitted shape736x512, proving real
M=16*46=736 (not padded4096). BF16-CK prefill selector requires M>=8192 and is
NOT responsible in this fixture. Restarted same configuration with rows736:
current diagnostic server2878897, log
/tmp/dsv4_tp8_slot736_diagnostic_20260908.log.
Client22373 completed one homogeneous32 wave and stage analysis. Dump directory
/tmp/dsv4_tp8_slot_layer0_20260908. Native AR/1M pool/all accepted flags retained.
Remove DEBUG_STAGE/DEBUG_ATTN env before reporting future performance.

All16prefixes validated using dumped input IDs/positions, starts0,46,...690.
Layer0 row comparisons (736valid rows):
- attn_residual, attn_pre_norm, attn_norm:736exact, max_abs0.
- Q after prepare:538exact, max_abs0.03125.
- attention core:536exact, max_abs0.03125.
- wo_a:426exact, max_abs0.015625.
- wo_b partial:393exact; globalwo_b:260exact,max_abs0.03125.
- FFN out:252exact,max_abs0.0625.

This localizes first observed difference after normalized attention input,
before attention core. It is not an initial MHC/router/Mori error. Dumped
wqkv_a BF16 weight1536x4096 and normalized736x4096 input replay on GPU4:
ordinary torch BF16 linear yields617/736 identical rows,max_delta0.0009765625.
Ordinary FP32 linear followed byBF16 still yields only646exact,same maxdelta.
This demonstrates projection reduction/layout numerics can introduce the
observed slot dependence even with identical normalized inputs; it does not
prove there are no additional causes in later operators.

Independent fixed-K64 Triton projection oracle (BM16/BN64/4warps) yields736/736
exact,100/100 repeated-input mutations invariant. Relative to a first46-row
FP64 matmul roundedBF16, max_abs0.000244140625. No production dispatch wired.
Five graphABBA cycles:torch89.423us,candidate210.814us. Candidate is a numerical
repair oracle, not an acceptable performance replacement yet. Next: tune
fixed-order tiles without changing K reduction order, then test all subsequent
Q projections and E2E on identical-prefix fixtures. Do not simply extend the
old shared-gate fusion shape guard or blame the>=8192 CK branch.

Scripts:check_dsv4_identical_prefix_stages.py and
bench_dsv4_slot_stable_projection.py. Existing graph/correctness baseline
remains in3029524f14; no claim of completed repair or improved E2E yet.

## Fixed-order tile screen

Extended isolated oracle with BM/BN/BK/warps/stages parameters. Ten candidates
on idle physicalGPU4 while diagnostic service had no requests; amd-smi PIDs
all belonged to that service. No extra resident weight cache or model changes.
Best BM64/BN64/BK64/W4:112.966us versus pairedtorch89.636us, down from initial
210.814us but still26%slower. All ten candidates preserve736/736 row equality
on real input and100/100 repeated-input mutations. FP64-rounded reference
maximum error0.000244140625 remains unchanged on the real46-row subset.
BK128/256 did not help; best of these130.301us. Full candidate medians and
numeric witnesses in adjacent dsv4_slot_projection_tiles_20260908.json.

Not a completed fix: slot-stable wqkv_a alone cannot prove downstream Q/wo
projections or fullmodel invariance. Do not enable a global replacement based
on this isolated fixture. Next plausible implementation should retain a tuned
MFMA pipeline (e.g. explicit stable CK/BLAS solution) and verify identical-row
contract, rather than indiscriminately increasing Triton tile sizes.
