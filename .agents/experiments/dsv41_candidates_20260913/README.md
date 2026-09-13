# V4.1 candidate-block evidence

- `component.json`: GPU masks vs local official function, four lengths around
  and above16384, fixed replay and score/position mutations.
- `cache-prefix-summary.json`: cached vs recomputed top1 agreement at9positions;
  logprobs need not agree between these different execution modes.
- `recompute-regression.json`: before/after candidate patch, same full prefixes,
  output IDs and returned logprob arrays agree at all9positions.
- `fresh192-regression.json`: before/after192-token generation arrays agree;
  exact input is preserved in sibling `dsv41_indexer_stability_20260913/repeat192.json`.
- `france.json`: actual returned answer and completion IDs, natural EOS.
- `c4-summary.json`, `c4-warm-summary.json`:20semantic requests across5rounds;
  SQL formatting varies. `c4-round0-sql.json` and `c4-round1-sql.json` preserve
  both variants. Semantic pass is NOT cross-round token equality.

Full-model service uses8192token pool; the32769 component fixture does NOT
establish that full-model context capacity. See the associated memory note
for source hashes, launch provenance, limitations and next gates.
