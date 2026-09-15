# Post-query-owner current profile (2026-09-15)

Accepted performance stays7953.956103 input tok/s; original V4/TP8/native AR,
original weights,1M logical KV,32K chunk. Fresh same-profile service collected
128 rank-forward frames and64 exact input echoes with zero cache hits, then
stopped. No dense CPU tensor dump / CK replay hooks were enabled.

Three warm instrumented client waves16.50461/16.50812/16.51065s. Mean GPU envelope
**16.417479s**, using the longest rank envelope per forward and all stages from
that same rank. Fully closed, no independent-rank-max summation. Event/realtime
ratios within~0.003%; rank envelope spread at most4.5312ms.

Remaining nested budgets: routed MoE4.3775s, main attention2.7618s, two pre-mix
1.7303s, two post1.3688s, weighted norm0.5712s, Sinkhorn0.0655s. Entire MHC/norm
coarse boundaries total3.7631s. Attention output projection/collective1.5483s;
MoE output collective0.7474s is already inside MoE/collective5.3738s.

**Index-owner chain now0.3017s**, already including packing, logits, Top-K,
integer gather and physical reconstruction. Query producer remains0.6273s;
index weights0.0280s and index compressor0.1171s. Old pre-owner logits1.405s +
Top-K0.137s cannot be reused as remaining optimization budget. The new0.000184s
tail-marker gap is NOT actual Top-K runtime.

Full report/evidence directory `.agents/experiments/dsv4_c16_final_profile_20260915/`:
`current-owner.md`, `capture-owner-current/{analysis,details-analysis,plan}.json`,
`capture-owner-current-evidence.tar.gz` and its hash manifest. `capture_owner.py`
validates actual owner and pair hits on all ranks plus explicit profile flags.
Analysis now selects labels using `indexer_owner_chain_done`, preserving the
historical local-path labels. Seven CPU owner/label tests passed.

Next isolated screen is3/4 independent FP32 output columns sharing the same
eight activation rows, versus accepted2-column pre-mix. Not a merged BLOCK_N
reduction: exact K1024 chunk order stays independent. Must establish bitwise
correctness and a full-component gain before integrating. Production unchanged.
