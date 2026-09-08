#!/usr/bin/env python3
"""GPU exact public-wrapper A/B, standalone on one GCD."""
import json
import torch
from bench_dsv4_gfx90a_occupancy_bucket_oracle import make_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import gfx90a_fp4_expert_down_grouped

torch.manual_seed(2090801)
x=torch.randint(-127,128,(32,6,256),device='cuda',dtype=torch.int8)
s=torch.rand((32,6,8),device='cuda')*.01
w=torch.randint(0,256,(256,4096,128),device='cuda',dtype=torch.uint8)
ws=torch.randint(122,128,(256,4096,8),device='cuda',dtype=torch.uint8)
rw=torch.rand((32,6),device='cuda')
ids=torch.rand((32,256),device='cuda').topk(6,dim=1).indices.int()
meta=make_metadata(ids,assignments=4)
original_experts=meta.sorted_experts.clone()
graphs=[];outputs=[]
for enabled in (False,True):
    def run():
        return gfx90a_fp4_expert_down_grouped(x,s,w,ws,meta.sorted_ids,
            meta.sorted_experts,meta.valid,rw,assignments=4,rows=2,waves=8,
            blocks=832,use_lds_lut=True,uniform_metadata=enabled)
    run();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=run()
    graphs.append(g);outputs.append(out)
for iteration in range(100):
    x.random_(-127,128);s.uniform_(.001,.02);rw.uniform_(0,1)
    ws.random_(122,128)
    # Bijective expert relabeling changes metadata without changing buffer size.
    permutation=torch.randperm(256,device='cuda',dtype=torch.int32)
    meta.sorted_experts.copy_(permutation[original_experts.long()])
    for g in graphs:g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(outputs[1]).all()
    assert torch.equal(outputs[0].view(torch.int16),outputs[1].view(torch.int16)),iteration
    expected=outputs[1].clone()
    for _ in range(10):graphs[1].replay()
    torch.cuda.synchronize()
    assert torch.equal(expected.view(torch.int16),outputs[1].view(torch.int16)),iteration
print(json.dumps(dict(public_wrapper_exact_mutations=100,stable_replays=1000,
                      signed_zero_bits_checked=True,metadata_relabeling=True)))
