"""Single-GCD full-boundary oracle for opt-in config20 prefill iterations.

A: current env8, no prefill policy. B: same env8 + prefill config20 scope.
R: old policy with global env20. Geometry, Fn dtype and input tensors agree.
B/R must be byte-exact. This is not TP8/E2E or a semantic quality score.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sizes',type=int,nargs='+',default=[1,128,8192,32767])
p.add_argument('--mutations',type=int,default=10)
args=p.parse_args()
assert not args.output.exists() and args.mutations>0
assert all(1<=m<=65536 for m in args.sizes)
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
active={x['process_info']['pid'] for g in owners for x in g.get('process_list',[])
        if isinstance(x.get('process_info'),dict)}
assert active<={os.getpid()},f'GPU already occupied: {active}'

# Reuse the audited settings literal, not the old script's executable body.
import ast
root=Path(__file__).resolve().parent
repo=root.parents[2]
old=root.parent/'dsv4_prefill_mhc_priority_20260915/oracle.py'
tree=ast.parse(old.read_text())
settings=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
                              and any(isinstance(t,ast.Name) and t.id=='settings' for t in n.targets)))
settings['SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS']='1'
os.environ.update(settings)
import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.dsv4_prefill_experiments import _mix_reuse,_post_reuse
from sglang.srt.layers.dsv4_prefill_mhc_policy import _config_iters

torch.set_grad_enabled(False)
torch.manual_seed(20260915)
capture=root.parent/'dsv4_input_identity_20260914/trace-B1'
names=['ffn_mhc_residual','hc_ffn_fn','hc_ffn_scale','hc_ffn_base','ffn_norm_weight']
paths={n:capture/f'layer_0_rank_0_{n}.pt' for n in names}
t={n:torch.load(path,map_location='cuda',weights_only=True).contiguous() for n,path in paths.items()}
seed=t['ffn_mhc_residual'];fn=t['hc_ffn_fn'].float().contiguous();fn16=fn.half()
scale=t['hc_ffn_scale'].float();base=t['hc_ffn_base'].float();norm=t['ffn_norm_weight'].bfloat16()
assert seed.shape[1:]==(4,4096) and fn.shape==(24,16384)
original_group=mhc.get_tp_group;original_symmetric=mhc.is_allocation_symmetric
result=dict(status='running',scope=__doc__,settings=settings,results=[],
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    captured_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
    source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        repo/'python/sglang/kernels/ops/layernorm/mhc.py',
        repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_pre.py',
        repo/'python/sglang/srt/layers/dsv4_prefill_mhc_policy.py',
        repo/'python/sglang/srt/model_executor/runner/eager_runner.py')})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
def exact(a,b):return all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(a,b,strict=True))
def delta(a,b):return [float((x.float()-y.float()).abs().max()) for x,y in zip(a,b,strict=True)]
save()

for m in args.sizes:
    residual=seed.repeat((m+len(seed)-1)//len(seed),1,1)[:m].contiguous()
    x=torch.zeros((m,4096),device='cuda',dtype=torch.bfloat16)
    post=torch.zeros((m,4),device='cuda',dtype=torch.float32)
    comb=torch.eye(4,device='cuda').expand(m,4,4).contiguous()
    for bs in ([1] if m==1 else [1,2]):
        def call(arm):
            env_key='SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS'
            previous=os.environ[env_key]
            os.environ[env_key]='20' if arm=='R' else '8'
            policy_token=_config_iters.set(20 if arm=='B' else None)
            mix_token=_mix_reuse.set(8192<=m<=65536)
            post_token=_post_reuse.set(8192<=m<=65536)
            mhc.get_tp_group=lambda:None
            mhc.is_allocation_symmetric=lambda:False
            try:
                assert mhc.envs.SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS.get()==(20 if arm=='R' else 8)
                return mhc.mhc_fused_post_pre(x,residual,post,comb,fn,scale,base,
                    1e-6,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-6,
                    global_batch_size=bs,fn_fp16=fn16)
            finally:
                mhc.get_tp_group=original_group
                mhc.is_allocation_symmetric=original_symmetric
                _post_reuse.reset(post_token);_mix_reuse.reset(mix_token)
                _config_iters.reset(policy_token);os.environ[env_key]=previous
        for _ in range(3):
            for arm in ('A','B','R'):call(arm)
        a,b,r=call('A'),call('B'),call('R')
        result['last_reference_check']=dict(m=m,batch=bs,matched=exact(b,r),max_abs=delta(b,r));save()
        assert exact(b,r),(m,bs,'config20 differs from global20 reference')
        if bs==2:assert exact(a,b),'Already20 batch changed'
        initial=delta(a,b)
        samples={'A':[],'B':[]}
        for _ in range(3):
            for arm in ('A','B','B','A'):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):call(arm)
                end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)/5)
        repeats={}
        for arm in ('A','B'):
            ref=[v.clone() for v in call(arm)]
            repeats[arm]=sum(exact(ref,call(arm)) for _ in range(10))
            assert repeats[arm]==10,(m,bs,arm,'unstable component')
        for mutation in range(args.mutations):
            x.normal_(std=.1);post.normal_(std=.01)
            comb.copy_(torch.eye(4,device='cuda')[None,:,:]+torch.randn_like(comb)*.01)
            b,r=call('B'),call('R')
            assert exact(b,r),(m,bs,mutation)
            assert all(torch.isfinite(v).all() for v in b)
        reference=[v.clone() for v in b]
        order=torch.randperm(m,device='cuda');inverse=torch.argsort(order)
        for value in (x,residual,post,comb):value.copy_(value[order])
        assert exact(reference,[v[inverse] for v in call('B')]),(m,bs,'row permutation')
        item=dict(m=m,batch=bs,mutations=args.mutations,reference_exact=True,
            row_permutation_exact=True,replays_exact=repeats,initial_A_B_max_abs=initial,
            median_ms={k:statistics.median(v) for k,v in samples.items()},samples_ms=samples)
        result['results'].append(item);save();print(json.dumps({k:v for k,v in item.items() if k!='samples_ms'}),flush=True)
        del a,b,r,reference,ref,order,inverse
    del x,residual,post,comb
    torch.cuda.empty_cache()
result['status']='complete';save()
