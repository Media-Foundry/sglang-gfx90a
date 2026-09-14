"""Whole MHC boundary screen using captured residual/Fn and synthetic post inputs.

A preserves the launcher legacy batch1/FP16 split-K path. B suppresses only
legacy admission in an active large-prefill scope. R uses existing batch2
FP32 dispatch (20 Sinkhorn iterations); R8 aligns only its Sinkhorn iteration
selection with batch1 (8). B/R8 must be exact. R remains a diagnostic.
This does not modify production code or weights.
Run only after the service experiment stops; timings are not E2E throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sizes',type=int,nargs='+',default=[1,8192,32767,32768])
p.add_argument('--mutations',type=int,default=10)
p.add_argument('--replays',type=int,default=6)
args=p.parse_args()
assert not args.output.exists() and args.mutations>0 and args.replays>0
assert all(1<=m<=65536 for m in args.sizes)
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
active={x['process_info']['pid'] for g in owners for x in g.get('process_list',[]) if isinstance(x.get('process_info'),dict)}
assert active<={os.getpid()},f'GPU service or another experiment active: {active}'

settings={
    'SGLANG_DSV4_PREFILL_MIX_REUSE4':'1',
    'SGLANG_DSV4_PREFILL_POST_FUSED4':'1',
    'SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE':'8',
    'SGLANG_DSV4_GFX90A_MHC_BLOCK_K':'1024',
    'SGLANG_DSV4_GFX90A_MHC_TP_ONLY_GEOMETRY':'1',
    'SGLANG_DSV4_GFX90A_FUSE_MHC_POST_RMS_PARTIALS':'1',
    'SGLANG_DSV4_GFX90A_SPLITK_MHC_PRE_MIX':'1',
    'SGLANG_DSV4_GFX90A_FUSED_MHC_SPLITK_TAIL':'1',
    'SGLANG_DSV4_GFX90A_FP16_MHC_DOT':'1',
    'SGLANG_DSV4_GFX90A_BF16_MHC_DOT':'0',
    'SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE':'0',
    'SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE_FULL':'0',
    'SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS':'8',
    'SGLANG_DSV4_GFX90A_NATIVE_MHC_SINKHORN':'1',
    'SGLANG_DSV4_GFX90A_FUSED_MHC_WEIGHTED_RMS':'1',
    'SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION':'0',
}
os.environ.update(settings)
import torch
import triton
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.dsv4_prefill_experiments import _mix_reuse, _post_reuse

torch.manual_seed(20260915)
torch.set_grad_enabled(False)
root=Path(__file__).resolve().parent
repo=root.parents[2]
capture=root.parent/'dsv4_input_identity_20260914/trace-B1'
names=['ffn_mhc_residual','hc_ffn_fn','hc_ffn_scale','hc_ffn_base','ffn_norm_weight']
paths={name:capture/f'layer_0_rank_0_{name}.pt' for name in names}
tensors={name:torch.load(path,map_location='cuda',weights_only=True).contiguous() for name,path in paths.items()}
seed=tensors['ffn_mhc_residual']
fn=tensors['hc_ffn_fn'].float().contiguous()
scale=tensors['hc_ffn_scale'].float().contiguous()
base=tensors['hc_ffn_base'].float().contiguous()
norm=tensors['ffn_norm_weight'].bfloat16().contiguous()
assert seed.ndim==3 and seed.shape[1:]==(4,4096) and seed.dtype==torch.bfloat16
assert fn.shape==(24,16384) and scale.shape==(3,) and base.shape==(24,) and norm.shape==(4096,)
fn16=fn.half().contiguous()
original_admitted=mhc._mhc_fusion_admitted
original_tp_group=mhc.get_tp_group
original_symmetric=mhc.is_allocation_symmetric
original_sinkhorn=mhc.hc_split_sinkhorn
result=dict(status='running',scope=__doc__,settings=settings,results=[],
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    large_prefill_sinkhorn_iterations=dict(A=8,B=8,R=20,R8=8),
    allocation_scope='Single-GCD component with symmetric allocation disabled in all arms; not a TP8 collective/mempool benchmark.',
    captured_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
    source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in (
        'python/sglang/kernels/ops/layernorm/mhc.py',
        'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse.py',
        'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse8.py')})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
save()

def exact(a,b):
    return all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(a,b,strict=True))

def errors(a,b):
    result=[]
    for x,y in zip(a,b,strict=True):
        same=torch.equal(x.view(torch.uint8),y.view(torch.uint8))
        result.append(dict(bits_exact=same,max_abs=0. if same else float((x.float()-y.float()).abs().max()),
                           finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())))
    return result

def replay_stability(call, arm, count):
    # Clone: an implementation may return shared workspace. Comparing two
    # aliases after replay could otherwise report false determinism.
    reference=[t.clone() for t in call(arm)]
    comparisons=[]
    for _ in range(count):
        current=call(arm)
        differences=errors(reference,current)
        assert all(x['finite'] for x in differences),(arm,'non-finite replay')
        comparisons.append(differences)
    return dict(replays=count,
                exact_replays=sum(all(x['bits_exact'] for x in row) for row in comparisons),
                comparisons=comparisons)

for m in args.sizes:
    scope=8192<=m<=65536
    residual=seed.repeat(triton.cdiv(m,len(seed)),1,1)[:m].contiguous()
    x=torch.zeros(m,4096,device='cuda',dtype=torch.bfloat16)
    post=torch.zeros(m,4,device='cuda',dtype=torch.float32)
    comb=torch.eye(4,device='cuda').expand(m,4,4).contiguous()
    def call(arm):
        mix_token=_mix_reuse.set(scope)
        post_token=_post_reuse.set(scope)
        def admitted(bs):
            if arm=='B' and scope and _mix_reuse.get():return False
            return original_admitted(bs)
        mhc._mhc_fusion_admitted=admitted
        # The boundary launches no collective, but evaluates get_tp_group()
        # even for a disabled symmetric-allocation context. Match existing
        # single-GCD MHC oracles without pretending to initialize TP8.
        mhc.get_tp_group=lambda:None
        mhc.is_allocation_symmetric=lambda:False
        def aligned_sinkhorn(mixes,hc_scale,hc_base,hc_mult=4,sinkhorn_iters=20,
                             eps=1e-6,gfx90a_global_batch_size=None):
            return original_sinkhorn(mixes,hc_scale,hc_base,hc_mult,sinkhorn_iters,eps,1)
        if arm=='R8' and scope:mhc.hc_split_sinkhorn=aligned_sinkhorn
        try:
            return mhc.mhc_fused_post_pre(x,residual,post,comb,fn,scale,base,
                # The model-level argument is20; the gfx90a path uses the
                # existing environment override8 internally in both arms.
                1e-6,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-6,
                global_batch_size=2 if arm in ('R','R8') and scope else 1,fn_fp16=fn16)
        finally:
            mhc._mhc_fusion_admitted=original_admitted
            mhc.get_tp_group=original_tp_group
            mhc.is_allocation_symmetric=original_symmetric
            mhc.hc_split_sinkhorn=original_sinkhorn
            _post_reuse.reset(post_token)
            _mix_reuse.reset(mix_token)
    for _ in range(3):
        for arm in ('A','B','R','R8'):call(arm)
    a,b,r,r8=call('A'),call('B'),call('R'),call('R8')
    result['last_reference_check']=dict(m=m,matched8=errors(b,r8),batch2_20=errors(b,r))
    save()
    assert exact(b,r8),'Proposed priority differs from matched-iteration FP32 reference'
    reference_20_difference=errors(b,r)
    if not scope:assert exact(a,b),'Small-M legacy behavior changed'
    initial=errors(a,b)
    assert all(x['finite'] for x in initial)
    stability={arm:replay_stability(call,arm,args.replays) for arm in ('A','B')}
    assert stability['B']['exact_replays']==args.replays,'Candidate is not replay-stable'
    samples={'A':[],'B':[]}
    for _ in range(3):
        for arm in ('A','B','B','A'):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):call(arm)
            end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)/5)
    mutation_differences=[]
    for mutation in range(args.mutations):
        residual.mul_(torch.empty(m,1,1,device='cuda',dtype=torch.bfloat16).uniform_(.97,1.03))
        x.normal_(std=.1);post.normal_(std=.01)
        comb.copy_(torch.eye(4,device='cuda')[None,:,:]+torch.randn_like(comb)*.01)
        a,b,r8=call('A'),call('B'),call('R8')
        assert exact(b,r8),(m,mutation,'matched-iteration FP32 reference')
        if not scope:assert exact(a,b),(m,mutation,'small M')
        differences=errors(a,b)
        assert all(x['finite'] for x in differences),(m,mutation,'non-finite A/B')
        mutation_differences.append(differences)
    original=[t.clone() for t in b]
    order=torch.randperm(m,device='cuda');inverse=torch.argsort(order)
    for t in (residual,x,post,comb):t.copy_(t[order])
    permuted=call('B')
    assert exact(original,[t[inverse] for t in permuted]),'Candidate row permutation'
    item=dict(m=m,active_large_prefill=scope,reference_exact=True,mutations=args.mutations,
              row_permutation_exact=True,legacy_difference=initial,
              batch2_20_iteration_difference=reference_20_difference,
              mutation_differences=mutation_differences,replay_stability=stability,samples_ms=samples,
              median_ms={k:median(v) for k,v in samples.items()})
    item['speedup']=item['median_ms']['A']/item['median_ms']['B']
    result['results'].append(item);save();print(json.dumps(item),flush=True)
    del residual,x,post,comb,a,b,r,r8,original,permuted,order,inverse
    torch.cuda.empty_cache()
result['status']='complete';save()
