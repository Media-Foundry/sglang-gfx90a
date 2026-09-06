"""Compare logical Top-K sets, masking unused logit capacity."""
import argparse
import json
from pathlib import Path

import torch


def logical_ids(d):
    physical = d["indices"].long()
    matches = physical[:, :, None] // 64 == d["page_table"].long()[:, None, :]
    assert (matches.sum(-1)[physical >= 0] == 1).all()
    result = matches.int().argmax(-1) * 64 + physical % 64
    return result.masked_fill(physical < 0, -1)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("--replay", type=int, default=1)
args = parser.parse_args()
for layer in range(2, 43, 2):
    paths = [args.root / f"layer_{layer}_replay_{rep}.pt" for rep in (0, args.replay)]
    if not all(p.exists() for p in paths):
        continue
    a, b = [torch.load(p, weights_only=True, map_location="cpu") for p in paths]
    # Trivial <=512 rows may deliberately skip score computation entirely.
    mask = ((torch.arange(a["logits"].shape[1])[None, :] < a["seq_lens"][:, None])
            & (a["seq_lens"][:, None] > 512))
    lhs, rhs = logical_ids(a), logical_ids(b)
    set_changed = (lhs.sort(-1).values != rhs.sort(-1).values).any(-1)
    scores = a["logits"].masked_fill(~mask, -float("inf")).topk(513, dim=-1).values
    ties = (scores[:, 511] == scores[:, 512]) & (a["seq_lens"] > 512)
    print(json.dumps(dict(layer=layer,
                          valid_logits_exact=torch.equal(a["logits"][mask], b["logits"][mask]),
                          logical_order_exact=torch.equal(lhs, rhs),
                          changed_set_rows=set_changed.nonzero().flatten().tolist(),
                          cutoff_tie_rows=ties.nonzero().flatten().tolist())), flush=True)
    if set_changed.any() or not torch.equal(a["logits"][mask], b["logits"][mask]):
        break
