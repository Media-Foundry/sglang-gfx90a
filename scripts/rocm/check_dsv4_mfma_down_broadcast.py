"""Compare identical MFMA math with direct versus wave-broadcast operands."""
import argparse
import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import gfx90a_fp4_expert_down_mfma32

p = argparse.ArgumentParser()
p.add_argument('--mutations', type=int, default=20)
args = p.parse_args()
torch.manual_seed(12345)
e, m, t, n, k = 256, 2304, 6, 64, 512
w = torch.randint(0, 256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
s = shuffle_scale_a16w4(torch.full((e*n, k//32), 115, device='cuda', dtype=torch.uint8), e, False)
x = torch.randint(-127, 128, (m,t,k), device='cuda', dtype=torch.int8)
xs = torch.rand((m,t,k//32), device='cuda') * .01
weights = torch.rand((m,t), device='cuda')
failures = 0
for assignments in (32,64):
    for mutation in range(args.mutations):
        ids = torch.rand((m,e), device='cuda').topk(t,dim=1).indices.int()
        si, _, se, valid, _ = moe_sorting(ids, weights, e, n, torch.bfloat16, block_size=assignments)
        weights.uniform_()
        outputs = [gfx90a_fp4_expert_down_mfma32(x,xs,w,s,si,se,valid,weights,
                   blocks=32,split=2,broadcast_scales=b,assignments=assignments)
                   for b in (0,1)]
        torch.cuda.synchronize()
        a,b = outputs
        equal = torch.equal(a,b)
        failures += not equal
        print(dict(assignments=assignments,mutation=mutation,exact=equal,
                   mismatches=(a!=b).sum().item(),max_abs=(a.float()-b.float()).abs().max().item()), flush=True)
print(f'failures={failures}', flush=True)
raise SystemExit(1 if failures else 0)
