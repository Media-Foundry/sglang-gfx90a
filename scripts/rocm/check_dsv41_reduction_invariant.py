"""Real-fixture and mutation/graph checks for the two V4.1 row reductions."""

import argparse
import json
from pathlib import Path

import torch

from compare_dsv41_row_trace import row_metrics
from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.layernorm.dsv41_invariant import hc_mix_stats_invariant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mutations', type=int, default=100)
    p.add_argument('--replays', type=int, default=1000)
    args = p.parse_args()
    assert args.mutations > 0 and args.replays > 0
    results, examples = [], {}
    torch.manual_seed(20260913)
    for file in sorted(args.trace_dir.glob('*.pt')):
        r = torch.load(file, weights_only=True, map_location='cpu')
        tag = r['tag']
        if (tag == '_hc_mix_and_combine' and r['call'] not in (2,3)) or (tag == 'engram_gate' and r['call'] != 1) or tag == 'hc_post':
            continue
        data = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in r['args'].items()}
        examples[tag] = data
        one = run(tag,data)
        x = data['x'].double()
        if tag == '_hc_mix_and_combine':
            x = x.flatten(1)
            reference = ((x @ data['hc_fn'].double().t()) * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-20)).float()
            torch.testing.assert_close(one,reference,atol=5e-6,rtol=2e-5)
        else:
            k,v = data['kv'].double().split([20480,5120],-1)
            k = k.unflatten(-1,(4,5120))
            dot = (x*data['q_weight'].double()*data['k_weight'].double()*k).sum(-1)
            dot *= torch.rsqrt(x.square().mean(-1)+data['eps'])*torch.rsqrt(k.square().mean(-1)+data['eps'])*5120**-0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(data['clamp_value']).sqrt(),dot))
            reference = (x+gate.unsqueeze(-1)*v.unsqueeze(-2)).to(data['x'].dtype)
            torch.testing.assert_close(one,reference,atol=1e-6,rtol=0.015625)
        for m in (2,3,15,16,17,32,64,128,203,204,256):
            many = expand(data,m)
            # Surround the fixed row with other real input values, not only
            # a repeated prompt. Test first/middle/last row placement.
            many['x'] = (many['x'].float()*(1+torch.randn_like(many['x'].float())*.03)).to(many['x'].dtype)
            for pos in sorted({0,m//2,m-1}):
                many['x'][pos].copy_(data['x'][0])
                out = run(tag,many)
                assert torch.equal(one[0],out[pos]), (file,m,pos)
        results.append({'fixture':file.name,'tag':tag,'vs_fp64':row_metrics(one.cpu(),reference.cpu()),'placement_checks':32})
        print(json.dumps(results[-1]),flush=True)
    assert len(examples)==2 and len(results)==12, 'incomplete real fixture coverage'
    for tag,data in examples.items():
        one_data = {**data,'x':data['x'].clone()}
        many = expand(data,17)
        for i in range(args.mutations):
            one_data['x'].copy_((data['x'].float()*(1+torch.randn_like(data['x'].float())*.05)).to(data['x'].dtype))
            if tag == 'engram_gate':
                one_data['kv'] = (data['kv'].float()*(1+torch.randn_like(data['kv'].float())*.05)).to(data['kv'].dtype)
                many['kv'].copy_(one_data['kv'].expand_as(many['kv']))
            many['x'].copy_(one_data['x'].expand_as(many['x']))
            y1,y17=run(tag,one_data),run(tag,many)
            assert torch.isfinite(y17).all() and torch.equal(y17,y1.expand_as(y17))
        graph=torch.cuda.CUDAGraph()
        run(tag,many)
        with torch.cuda.graph(graph):
            replay_out=run(tag,many)
        for i in range(args.replays):
            # Change inputs outside the graph and compare each replay to eager.
            if i % 10 == 0:
                many['x'].copy_((many['x'].float()*.999).to(many['x'].dtype))
            expected=run(tag,many)
            graph.replay()
            assert torch.equal(replay_out,expected), (tag,i)
        print(tag,'mutations',args.mutations,'graph replays',args.replays,'PASS',flush=True)
    with args.output.open('x') as f:
        json.dump({'fixtures':results,'mutations_per_component':args.mutations,
                   'graph_replays_per_component':args.replays},f,indent=2)


def expand(data,m):
    result=dict(data)
    for k in ('x','kv'):
        if k in result:
            v=result[k]
            assert v.shape[0]==1
            result[k]=v.expand(m,*v.shape[1:]).contiguous()
    return result


def run(tag,data):
    if tag == '_hc_mix_and_combine':
        return hc_mix_stats_invariant(data['x'].flatten(1),data['hc_fn'],1e-20)
    return fused_engram_gate(data['x'],data['kv'],data['q_weight'],data['k_weight'],data['eps'],data['clamp_value'])


if __name__=='__main__':
    with torch.no_grad():
        main()
