"""CPU FP64 dot reference at the four reproduced service discrepancies."""
import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
run = root / 'layer0-changed-rows'
def load(arm, stage):
    return torch.load(run / ('trace-' + arm) / ('layer_0_rank_0_' + stage + '.pt'),
                      weights_only=True)[-8192:]

xa, xb = (load(arm, 'prepare_full_input') for arm in ('A1', 'B1'))
ya, yb = (load(arm, 'prepare_full_qkv_a') for arm in ('A1', 'B1'))
assert torch.equal(xa, xb)
weight = torch.load(root / 'all-ranks/trace-A1/layer_0_rank_0_projection_wqkv_a.pt',
                    weights_only=True)
locations = (ya != yb).nonzero().tolist()
assert len(locations) == 4
records = []
for row, col in locations:
    reference = (xa[row].double() * weight[col].double()).sum()
    a, b = float(ya[row, col]), float(yb[row, col])
    midpoint = (a + b) / 2
    records.append(dict(position=row, column=col, a=a, b=b, fp64=float(reference),
                        fp64_rounded_bf16=float(reference.bfloat16()),
                        midpoint=midpoint, distance_to_midpoint=float(reference)-midpoint))
out = root / 'qkv-tie-analysis.json'
assert not out.exists()
out.write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2))
