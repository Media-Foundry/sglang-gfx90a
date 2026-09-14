"""PREPARED: full20 boundary versus full8 + comb-only12, after service shutdown.

Reuses current production kernels and captured residual/Fn; synthetic post
inputs. Require every returned tensor bit-exact before accepting any timing.
Only physical GPU4, no concurrent service. This is not an E2E result.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sizes',type=int,nargs='+',default=[1,128,8192,32767,32768])
p.add_argument('--mutations',type=int,default=10)
p.add_argument('--graph-replays',type=int,default=0,
               help='Optional fixed-input graph replays plus one input-mutation replay.')
p.add_argument('--mutate-parameters',action='store_true',
               help='Perturb local fixture Fn/scale/base and residual; never writes checkpoint weights.')
p.add_argument('--integrated',action='store_true',
               help='Use production config20 scope/refinement selector; small M must retain full20.')
args=p.parse_args()
assert not args.output.exists() and args.mutations>0 and args.graph_replays>=0
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
assert all(1<=m<=65536 for m in args.sizes)
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
active={x['process_info']['pid'] for g in owners for x in g.get('process_list',[])
        if isinstance(x.get('process_info'),dict)}
assert active<={os.getpid()},f'GPU occupied; do not overlap ABBA: {active}'

root=Path(__file__).resolve().parent;repo=root.parents[2]
old=root.parent/'dsv4_prefill_mhc_priority_20260915/oracle.py'
tree=ast.parse(old.read_text())
settings=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='settings' for t in n.targets)))
settings['SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS']=str(int(args.integrated))
settings['SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20']=str(int(args.integrated))
os.environ.update(settings)
import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.dsv4_prefill_experiments import _mix_reuse,_post_reuse
from sglang.srt.layers.dsv4_prefill_mhc_policy import _config_iters
if args.integrated:
    from sglang.kernels.ops.layernorm import gfx90a_mhc_comb_refine as production
    original_refine=production.refine_comb20
    refine_calls=[0]
    def counted_refine(*a,**k):
        refine_calls[0]+=1
        return original_refine(*a,**k)
    production.refine_comb20=counted_refine
spec=importlib.util.spec_from_file_location('refine20_candidate',root/'candidate.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
torch.set_grad_enabled(False);torch.manual_seed(20260915)
capture=root.parent/'dsv4_input_identity_20260914/trace-B1'
names=['ffn_mhc_residual','hc_ffn_fn','hc_ffn_scale','hc_ffn_base','ffn_norm_weight']
paths={n:capture/f'layer_0_rank_0_{n}.pt' for n in names}
t={n:torch.load(path,map_location='cuda',weights_only=True).contiguous() for n,path in paths.items()}
seed=t['ffn_mhc_residual'];fn=t['hc_ffn_fn'].float().contiguous();fn16=fn.half()
scale=t['hc_ffn_scale'].float();base=t['hc_ffn_base'].float();norm=t['ffn_norm_weight'].bfloat16()
initial_params=(fn.clone(),scale.clone(),base.clone()) if args.mutate_parameters else None
original_group=mhc.get_tp_group;original_symmetric=mhc.is_allocation_symmetric
result=dict(status='running',scope=__doc__,settings=settings,results=[],
    mutate_parameters=args.mutate_parameters,integrated=args.integrated,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),root/'candidate.py',repo/'python/sglang/kernels/ops/layernorm/mhc.py',
        repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_comb_refine.py',
        repo/'python/sglang/srt/layers/dsv4_prefill_mhc_policy.py')},
    captured_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
def exact(a,b):return all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(a,b,strict=True))
def delta(a,b):return [float((x.float()-y.float()).abs().max()) for x,y in zip(a,b,strict=True)]
save()
for m in args.sizes:
    residual=seed.repeat((m+len(seed)-1)//len(seed),1,1)[:m].contiguous()
    x=torch.zeros((m,4096),device='cuda',dtype=torch.bfloat16)
    post=torch.zeros((m,4),device='cuda',dtype=torch.float32)
    comb=torch.eye(4,device='cuda').expand(m,4,4).contiguous()
    def call(arm):
        key='SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS';previous=os.environ[key]
        os.environ[key]='20' if arm=='A' or args.integrated else '8'
        policy_token=_config_iters.set(20 if args.integrated and arm=='B' else None)
        before_refine=refine_calls[0] if args.integrated else 0
        mix_token=_mix_reuse.set(8192<=m<=65536)
        post_token=_post_reuse.set(8192<=m<=65536)
        mhc.get_tp_group=lambda:None;mhc.is_allocation_symmetric=lambda:False
        try:
            out=mhc.mhc_fused_post_pre(x,residual,post,comb,fn,scale,base,
                1e-6,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-6,
                global_batch_size=1,fn_fp16=fn16)
            if args.integrated:
                assert refine_calls[0]-before_refine==int(arm=='B' and 8192<=m<=65536)
            elif arm=='B':candidate.refine12[(m,)](out[2],eps=1e-6,num_warps=1)
            return out
        finally:
            mhc.get_tp_group=original_group;mhc.is_allocation_symmetric=original_symmetric
            _post_reuse.reset(post_token);_mix_reuse.reset(mix_token);os.environ[key]=previous
            _config_iters.reset(policy_token)
    for _ in range(3):call('A');call('B')
    a,b=call('A'),call('B')
    result['last_check']=dict(m=m,exact=exact(a,b),max_abs=delta(a,b));save()
    assert exact(a,b),(m,'full20 versus8+12',result['last_check'])
    for mutation in range(args.mutations):
        x.normal_(std=.1);post.normal_(std=.01)
        comb.copy_(torch.eye(4,device='cuda')[None,:,:]+torch.randn_like(comb)*.01)
        if args.mutate_parameters:
            # Different mix distributions, including near-saturated logits;
            # both arms consume exactly the same temporary fixture tensors.
            fn.copy_(initial_params[0]*(1+.1*torch.randn_like(fn)))
            fn16.copy_(fn)
            scale.copy_(initial_params[1]*(.5+torch.rand_like(scale)))
            base.copy_(initial_params[2]+torch.randn_like(base)*(.1,1.,4.,8.)[mutation%4])
            residual.normal_(std=.1)
        a,b=call('A'),call('B')
        result['last_check']=dict(m=m,mutation=mutation,exact=exact(a,b),max_abs=delta(a,b));save()
        assert exact(a,b),(m,mutation,'mismatch')
        assert all(torch.isfinite(v).all() for v in b)
    # Row ownership must not depend on batch order, including ragged M.
    original_inputs=(residual,x,post,comb)
    permutation=torch.randperm(m,device='cuda')
    residual,x,post,comb=(v.index_select(0,permutation) for v in original_inputs)
    permuted=call('B')
    expected=tuple(v.index_select(0,permutation) for v in a)
    result['last_check']=dict(m=m,phase='row_permutation',exact=exact(expected,permuted),
                              max_abs=delta(expected,permuted));save()
    assert exact(expected,permuted),(m,'row permutation mismatch')
    residual,x,post,comb=original_inputs
    del original_inputs,permutation,permuted,expected
    if args.graph_replays:
        graph=torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output=call('B')
        reference=call('A')
        for replay in range(args.graph_replays):
            graph.replay()
            assert exact(reference,graph_output),(m,'graph replay',replay)
        # A graph which simply reuses stale output must fail this check.
        x.normal_(std=.1);post.normal_(std=.01)
        if args.mutate_parameters:
            base.add_(.03125)
        reference=call('A')
        graph.replay()
        result['last_check']=dict(m=m,phase='graph_input_mutation',
            exact=exact(reference,graph_output),max_abs=delta(reference,graph_output));save()
        assert exact(reference,graph_output),(m,'graph stale input or mismatch')
        del graph,graph_output,reference
    samples={'A':[],'B':[]}
    for _ in range(3):
        for arm in ('A','B','B','A'):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):call(arm)
            end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)/5)
    item=dict(m=m,mutations=args.mutations,all_outputs_exact=True,row_permutation_exact=True,
              graph_replays=args.graph_replays,graph_mutation_exact=True if args.graph_replays else None,
              median_ms={k:statistics.median(v) for k,v in samples.items()},samples_ms=samples)
    result['results'].append(item);save();print(json.dumps(item),flush=True)
    del residual,x,post,comb,a,b
    torch.cuda.empty_cache()
result['status']='complete';save()
