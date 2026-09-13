"""Replay V4.1 router on captured full-prefill prefixes using raw gate weights.

This is not a captured runtime-router-output oracle: it checks the documented
BF16 weight / AIter dispatcher contract independently. Reference selections
below are not substituted for service Top-K observations.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/media/PM983/deepseek-v4.1-flash'))
    p.add_argument('--module-number', type=int, default=37)
    p.add_argument('--layer', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-invariant', action='store_true')
    args = p.parse_args()
    files=list(args.trace_dir.glob(f'*DeepseekV2MoE-module{args.module_number}-call0-*'))
    assert len(files)==1
    first=torch.load(files[0],weights_only=True,map_location='cpu')
    full=torch.load(str(files[0]).replace('-call0-','-call8-'),weights_only=True,map_location='cpu')
    assert first['layer_id']==full['layer_id']==args.layer
    x,y=first['args'][0].cuda(),full['args'][0].cuda()
    assert x.shape==(203,5120) and y.shape==(209,5120) and torch.equal(x,y[:203])
    index=json.loads((args.model_dir/'model.safetensors.index.json').read_text())['weight_map']
    name=f'layers.{args.layer}.ffn.gate.weight'
    with safe_open(args.model_dir/index[name],framework='pt',device='cpu') as f:
        weight=f.get_tensor(name)
        bias=f.get_tensor(f'layers.{args.layer}.ffn.gate.bias')
    assert weight.dtype==torch.bfloat16 and weight.shape==(384,5120)
    digest=hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
    weight=weight.cuda()
    a,b=aiter_dsv3_router_gemm(x,weight),aiter_dsv3_router_gemm(y,weight)
    counts=(a!=b[:203]).sum(-1)

    def reference_selection(logits):
        scores=torch.nn.functional.softplus(logits.float()).sqrt()
        chosen=(scores+bias.bfloat16().float().cuda()).argsort(descending=True,stable=True)[:6]
        w=scores[chosen];w=w/w.sum()*1.5
        return {'ids':chosen.tolist(),'weights':w.tolist()}

    rows=[]
    for row in (counts>0).nonzero().flatten().tolist():
        changed=(a[row]!=b[row]).nonzero().flatten().tolist()
        rows.append({'row':row,'logits':[{'expert':e,'M203':float(a[row,e]),'M209':float(b[row,e])} for e in changed],
                     'M203_reference_selection':reference_selection(a[row]),
                     'M209_reference_selection':reference_selection(b[row])})
    result={'raw_weight_sha256':digest,'raw_weight_dtype':'bfloat16','raw_bias_dtype':str(bias.dtype),
            'runtime_router_output_captured':False,'same_prefix_inputs':True,'changed_rows':rows}
    if args.check_invariant:
        from sglang.kernels.ops.moe.dsv41_router_invariant import router_bf16_invariant
        from compare_dsv41_row_trace import row_metrics
        ca,cb=router_bf16_invariant(x,weight),router_bf16_invariant(y,weight)
        assert torch.equal(ca,cb[:203])
        ref=(x[71:72].double() @ weight.double().t()).to(x.dtype)
        torch.testing.assert_close(ca[71:72],ref,atol=1e-5,rtol=0.015625)
        result['candidate']={'all_203_prefix_rows_exact':True,'row71_vs_fp64':row_metrics(ca[71:72].cpu(),ref.cpu())}
        torch.manual_seed(20260913)
        for i in range(100):
            row=(x[71:72].float()*(1+torch.randn_like(x[71:72].float())*.03)).to(x.dtype)
            one=router_bf16_invariant(row,weight)
            for m in (1,16,64,203,209):
                many=row.expand(m,-1).contiguous()
                out=router_bf16_invariant(many,weight)
                assert torch.equal(out,one.expand_as(out))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout=router_bf16_invariant(y,weight)
        for i in range(1000):
            if i%10==0:
                y.copy_((y.float()*.999).to(y.dtype))
            expected=router_bf16_invariant(y,weight)
            graph.replay()
            assert torch.equal(gout,expected)
        result['candidate'].update(mutations=100,shapes_per_mutation=5,graph_replays=1000)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with torch.no_grad():
        main()
