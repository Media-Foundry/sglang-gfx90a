"""Separate M-shape from row-offset effects using identical real input vectors."""
import json
import os
from pathlib import Path
import torch

assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
root = Path(__file__).resolve().parent
output = root / 'qkv-shape-row-axes.json'
assert not output.exists()
source = torch.load(root / 'layer0-changed-rows/trace-A1/layer_0_rank_0_prepare_full_input.pt',
                    weights_only=True)[-8192:].cuda()
weight = torch.load(root / 'all-ranks/trace-A1/layer_0_rank_0_projection_wqkv_a.pt',
                    weights_only=True).cuda()
positions = (512, 2048, 3328, 3840)
buffers = [torch.zeros((m, 4096), device='cuda', dtype=torch.bfloat16) for m in (32768,32767)]
for x in buffers:
    for p in positions:
        x[24576+p].copy_(source[p]); x[24575+p].copy_(source[p])
ys = [torch.nn.functional.linear(x, weight) for x in buffers]
def compare(a,b):
    return dict(changed=int(torch.count_nonzero(a!=b)),
                max_abs=float((a.float()-b.float()).abs().max()))
records=[]
for p in positions:
    a,b=24576+p,24575+p
    records.append(dict(position=p,
        row_shift_m32768=compare(ys[0][a],ys[0][b]),
        row_shift_m32767=compare(ys[1][a],ys[1][b]),
        shape_change_at_row_a=compare(ys[0][a],ys[1][a]),
        shape_change_at_row_b=compare(ys[0][b],ys[1][b])))
output.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
