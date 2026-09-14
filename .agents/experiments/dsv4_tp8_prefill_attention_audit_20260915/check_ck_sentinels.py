"""Pending GPU contract check for existing H8 CK; no production edits.

Q=0, valid KV rows=1, sink=0 gives output valid_count/(valid_count+1).
Negative slots must not add softmax mass. This is not a speed benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--candidate', action='store_true', help='Isolated experimental JIT; no production selector changes.')
p.add_argument('--stress', action='store_true', help='100 random mutations and1000 graph replays; candidate only.')
args=p.parse_args();assert not args.output.exists()
assert not args.stress or args.candidate
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
active={r['process_info']['pid'] for g in owners for r in g.get('process_list',[])
        if isinstance(r.get('process_info'),dict)}
assert active<={os.getpid()},f'GPU occupied: {active}'

import torch
from sglang.kernels.ops.attention.dsv4.gfx90a_sparse_h8 import _module
if args.candidate:
    from torch.utils.cpp_extension import include_paths
    from sglang.kernels.jit.utils import load_jit
    def _module():
        return load_jit(
            'gfx90a_dsv4_sparse_h8_sentinel_oracle_v1',
            cuda_files=['deepseek_v4/gfx90a_dsv4_sparse_h8_oracle.cuh'],
            cuda_wrappers=[('run', 'sglang::Gfx90aDsv4SparseH8Oracle::run')],
            extra_cuda_cflags=['-O3', '-std=c++20', '-DCK_ENABLE_BF16',
                              '-DCK_USE_XDL', '-DSGLANG_DSV4_CK_SENTINEL_ORACLE=1'],
            extra_include_paths=[*include_paths(),
                '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include',
                '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/library/include'])

repo=Path(__file__).resolve().parents[3]
sources=['python/sglang/kernels/ops/attention/dsv4/gfx90a_sparse_h8.py',
    'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_dsv4_sparse_h8_oracle.cuh',
    'python/sglang/kernels/jit/csrc/deepseek_v4/dsv4_unified_sparse_decode_ck.cuh']
result=dict(status='running',scope=__doc__,candidate=args.candidate,cases=[],
    source_sha256={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in sources})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
save()
q=torch.zeros((1,8,512),device='cuda',dtype=torch.bfloat16)
kv=torch.ones((2,512),device='cuda',dtype=torch.bfloat16)
sink=torch.zeros(8,device='cuda',dtype=torch.float32)
workspace=torch.empty(2*8*514*4,device='cuda',dtype=torch.uint8)
for name,slots in [('valid',[0]),('mixed_negative',[0,-1]),
                   ('other_negative',[-2,1]),('duplicate',[0,0]),
                   ('all_negative',[-1,-2]),('empty',[]),
                   ('out_of_range',[0,2]),('invalid_first_tile',[-1]*16+[0]),
                   ('invalid_middle_tile',[0]*16+[-1]*16+[1]*16),
                   ('pipeline_valid_to_invalid',[0]*16+[-1]*32+[1]),
                   ('pipeline_invalid_to_valid',[-1]*16+[0]*16+[-1]*16+[1]),
                   ('top512_mixed',([0,-1,1,-2]*128))]:
    indices=torch.tensor(slots or [-1],device='cuda',dtype=torch.int32)
    indptr=torch.tensor([0,len(slots)],device='cuda',dtype=torch.int32)
    output=torch.empty_like(q)
    _module().run(q,kv,indices,indptr,sink,output,workspace,512**-.5)
    count=sum(0<=s<kv.shape[0] for s in slots)
    reference=torch.full_like(q,count/(count+1))
    delta=float((output.float()-reference.float()).abs().max())
    exact=torch.equal(output.view(torch.uint8),reference.view(torch.uint8))
    result['cases'].append(dict(name=name,slots=slots,valid_occurrences=count,
        expected=float(reference.flatten()[0]),actual=float(output.flatten()[0]),
        all_finite=bool(torch.isfinite(output).all()),max_abs=delta,exact=exact))
    save();print(json.dumps({k:v for k,v in result['cases'][-1].items() if k!='slots'}),flush=True)
if args.stress:
    torch.manual_seed(20260915)
    module=_module()
    q=torch.empty((4,8,512),device='cuda',dtype=torch.bfloat16)
    kv=torch.empty((64,512),device='cuda',dtype=torch.bfloat16)
    sink=torch.empty(8,device='cuda',dtype=torch.float32)
    boundaries=[0,0,17,50,562]
    indptr=torch.tensor(boundaries,device='cuda',dtype=torch.int32)
    indices=torch.empty(boundaries[-1],device='cuda',dtype=torch.int32)
    output=torch.empty_like(q)
    workspace=torch.empty(4*2*8*514*4,device='cuda',dtype=torch.uint8)
    def mutate():
        q.normal_(std=.15);kv.normal_(std=.2);sink.uniform_(-2,2)
        indices.random_(-16,80)
    def run():
        module.run(q,kv,indices,indptr,sink,output,workspace,512**-.5)
    def reference():
        answer=[]
        for row,(start,end) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            slots=indices[start:end];slots=slots[(slots>=0)&(slots<64)]
            values=kv[slots.long()].float()
            scores=q[row].float()@values.T*(512**-.5)
            prob=torch.softmax(torch.cat((scores,sink[:,None]),dim=1),dim=1)[:,:-1]
            answer.append(prob@values)
        return torch.stack(answer)
    worst=0.
    for trial in range(100):
        mutate();run();expected=reference()
        worst=max(worst,float((output.float()-expected).abs().max()))
        torch.testing.assert_close(output.float(),expected,atol=.002,rtol=.01)
    mutate();run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    graph.replay();torch.cuda.synchronize();fixed=output.clone()
    for _ in range(1000):graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output,fixed)
    mutate();run();torch.cuda.synchronize();fresh=output.clone()
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(output,fresh)
    torch.testing.assert_close(output.float(),reference(),atol=.002,rtol=.01)
    result['stress']=dict(mutations=100,max_abs_fp32_reference=worst,
        atol=.002,rtol=.01,graph_replays=1000,graph_exact=True,
        changed_inputs_graph_exact=True,
        scope='Synthetic Q/K/sink/ragged IDs, finite inputs, not target logits or throughput.')
    save();print(json.dumps(result['stress']),flush=True)
result['status']='complete';result['contract_pass']=all(c['exact'] for c in result['cases']);save()
assert result['contract_pass'],'CK sentinel contract mismatch; results retained for diagnosis'
