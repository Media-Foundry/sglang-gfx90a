"""Two-bank analytic and mutation contracts; not a service speed or logits test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[])),owners

import torch
from oracle import Runner, reference, check_mapping, ROOT, REPO
torch.manual_seed(20260915)
files=[Path(__file__),ROOT/'oracle.py',ROOT/'dsv4_prefill_two_source_core.cuh',
       REPO/'python/sglang/kernels/jit/csrc/debug/gfx90a_prefill_two_source_ck_oracle.cuh',
       REPO/'python/sglang/kernels/jit/csrc/deepseek_v4/dsv4_unified_sparse_decode_ck.cuh']
result=dict(status='running',scope=__doc__,cases=[],mutations=[],
            sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
def ints(x):return torch.tensor(x,device='cuda',dtype=torch.int32)
save()
q=torch.zeros((1,8,512),device='cuda',dtype=torch.bfloat16)
pkv=torch.ones((2,512),device='cuda',dtype=torch.bfloat16)
ekv=torch.full((2,512),3.,device='cuda',dtype=torch.bfloat16)
sink=torch.zeros(8,device='cuda')
cases=[('empty',[],[]),('prefix',[0],[]),('extend',[],[0]),
       ('both_banks',[0],[0]),('duplicates',[0,0],[0]),
       ('invalid',[-1,4],[0,-1,7]),('tile_boundary',[0]*15,[0]*17),
       ('all_invalid',[-1]*17,[-2]*33),('swa',[],[0]*128),
       ('top512_swa',[0]*512,[0]*128)]
for name,pids,eids in cases:
    runner=Runner(q,pkv,ints(pids),ints([0,len(pids)]),ekv,ints(eids),ints([0,len(eids)]),sink)
    np=sum(0<=s<2 for s in pids);ne=sum(0<=s<2 for s in eids)
    expected=torch.full_like(q,(np+3*ne)/(np+ne+1))
    for splits in (1,2):
        actual=runner(splits)
        check_mapping(runner)
        exact=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
        item=dict(case=name,splits=splits,exact=exact,actual=float(actual.flatten()[0]),expected=float(expected.flatten()[0]))
        result['cases'].append(item);save();print(item,flush=True)
        assert exact,item

q=torch.empty((4,8,512),device='cuda',dtype=torch.bfloat16)
pkv=torch.empty((64,512),device='cuda',dtype=torch.bfloat16)
ekv=torch.empty((96,512),device='cuda',dtype=torch.bfloat16)
pi=ints([0]*530);pp=ints([0,0,1,18,530])
ei=ints([0]*146);ep=ints([0,0,1,129,146])
sink=torch.empty(8,device='cuda')
runner=Runner(q,pkv,pi,pp,ekv,ei,ep,sink)
def mutate():
    q.normal_(std=.3);pkv.normal_(std=.5);ekv.normal_(std=.8);sink.uniform_(-2,2)
    pi.random_(-4,68);ei.random_(-4,100)
    # Change well-formed ragged boundaries within the same static capacities.
    pl=[0,1,17,int(torch.randint(400,513,()).item())]
    el=[0,1,int(torch.randint(80,129,()).item()),17]
    pp.copy_(ints([0,*torch.tensor(pl).cumsum(0).tolist()]))
    ep.copy_(ints([0,*torch.tensor(el).cumsum(0).tolist()]))
for n in range(100):
    mutate();expected=reference(q,pkv,pi,pp,ekv,ei,ep,sink)
    item=dict(mutation=n,max_abs={})
    for splits in (1,2):
        actual=runner(splits);check_mapping(runner)
        error=float((actual.float()-expected).abs().max())
        item['max_abs'][splits]=error
        assert bool(torch.isfinite(actual).all())
        try:
            torch.testing.assert_close(actual.float(),expected,atol=.003,rtol=.02)
        except AssertionError:
            fixture=args.output.with_suffix('.failure.pt')
            assert not fixture.exists()
            torch.save({name:t.cpu() for name,t in zip(
                ('q','pkv','pi','pp','ekv','ei','ep','sink','actual','expected'),
                (*runner.inputs,actual,expected),strict=True)},fixture)
            result.update(status='failed_fp32_tolerance',failure_mutation=n,
                          failure_splits=splits,failure_fixture=str(fixture),
                          failure_max_abs=error)
            save()
            raise
    result['mutations'].append(item);save()
    if n%20==0:print(item,flush=True)
runner();torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):graph_output=runner()
snapshot=graph_output.clone()
for _ in range(1000):graph.replay()
assert torch.equal(graph_output.view(torch.uint8),snapshot.view(torch.uint8))
mutate();graph.replay();changed=graph_output.clone();check_mapping(runner)
eager=runner()
assert torch.equal(changed.view(torch.uint8),eager.view(torch.uint8))
torch.testing.assert_close(changed.float(),reference(q,pkv,pi,pp,ekv,ei,ep,sink),atol=.003,rtol=.02)
result.update(status='complete',graph_replays=1000,graph_mutation_exact=True,
              maximum_abs_error=max(v for r in result['mutations'] for v in r['max_abs'].values()))
save();print({k:v for k,v in result.items() if k not in ('cases','mutations','sources')},flush=True)
