"""Isolated FP32 MFMA oracle, not a production selector or E2E claim."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
assert os.environ.get('HIP_VISIBLE_DEVICES') == '5'
import torch
import triton
import triton.language as tl
from module import load, load_split, load_cooperative
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_wave import post_wave_module
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair

@triton.jit
def finish(raw, partials, out, M):
    r = tl.program_id(0)
    p = tl.arange(0, 64)
    sq = tl.sum(tl.load(partials + r * 64 + p), 0)
    inv = tl.rsqrt(sq / 16384 + 1.e-6)
    c = tl.arange(0, 32)
    v = tl.load(raw + r * 24 + c, c < 24, 0)
    tl.store(out + r * 24 + c, v * inv, c < 24)

@triton.jit
def reduce_split(parts, raw, M, S: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0,256)
    v = tl.full((256,),0,tl.float32)
    for s in tl.static_range(S):
        v += tl.load(parts + s * M * 24 + i, i < M * 24, 0)
    tl.store(raw + i,v,i < M * 24)

root = Path(__file__).resolve().parent
csplit = '--cooperative-split' in sys.argv
split = '--split' in sys.argv or csplit
padded = '--padded' in sys.argv
cooperative = '--cooperative' in sys.argv or padded or csplit
assert not (split and cooperative and not csplit)
target = root / ('cooperative-split-screen.json' if csplit else ('padded-screen.json' if padded else ('cooperative-screen.json' if cooperative else ('split-screen.json' if split else 'screen.json'))))
assert not target.exists(), target
fixture = root.parent / 'dsv4_input_identity_20260914/trace-B1'
paths = [fixture / f'layer_0_rank_0_{n}.pt' for n in
         ('attn_out', 'ffn_mhc_residual', 'ffn_mhc_post', 'ffn_mhc_comb', 'hc_ffn_fn')]
seed = [torch.load(p, map_location='cuda', weights_only=True) for p in paths]
mod, postmod = (load_cooperative() if cooperative else (load_split() if split else load())), post_wave_module()
import ctypes
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
pci = ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0) == 0
report = dict(status='running', hip_visible_devices='5', pci_bus=pci.value.decode(),
    scope='Repeated/scaled sampled real layer0 inputs; raw MFMA plus RMS finish, not live full-M or service',
    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [*paths, root/'premix.cuh', root/'split.cuh', root/'cooperative.cuh', root/'module.py', Path(__file__)]}, cases=[])
def save(): target.write_text(json.dumps(report, indent=2)+'\n')
def error(a, b):
    d = a.double()-b.double()
    return dict(max_abs=d.abs().max().item(), relative_l2=(d.norm()/b.double().norm()).item(),
                bit_exact=torch.equal(a.view(torch.int32), b.view(torch.int32)),
                finite=bool(torch.isfinite(a).all()))

torch.manual_seed(2026091603)
for m in (17, 8192, 32767):
    x, residual, post, comb = [s.repeat(triton.cdiv(m,len(s)), *([1]*(s.ndim-1)))[:m].contiguous() for s in seed[:4]]
    post, comb, fn = post.view(m,4), comb.view(m,4,4), seed[4].contiguous()
    hidden, partials = torch.empty_like(residual), torch.empty((m,64),device='cuda')
    postmod.run(x,residual,post,comb,hidden,partials)
    flat = hidden.view(m,16384)
    a = torch.empty((m,24),device='cuda'); b = torch.empty_like(a); raw = torch.empty_like(a)
    def baseline(): premix8_pair[(12,triton.cdiv(m,8))](hidden,fn,partials,a,m,1.e-6,num_warps=1)
    case = dict(m=m, extra_scratch_bytes=raw.numel()*4, candidates=[])
    report['cases'].append(case); save()
    for name in (('s4','s16') if csplit else (('p32','p64','p128') if padded else (('k32','k64','k128') if cooperative else (('s4','s16','s32') if split else ('map0','u4','u8'))))):
        if split:
            s = int(name[1:])
            scratch = torch.empty((s,m,24),device='cuda')
            def op(h,f,r):
                getattr(mod,name)(h,f,scratch)
                reduce_split[(triton.cdiv(m*24,256),)](scratch,r,m,s,num_warps=4)
        else:
            op = getattr(mod,name)
        def candidate():
            op(flat,fn,raw)
            finish[(m,)](raw,partials,b,m,num_warps=1)
        checks=[]
        for mutation in range(5):
            if mutation:
                x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
                postmod.run(x,residual,post,comb,hidden,partials)
            b.fill_(float('nan')); baseline(); candidate()
            checks.append(error(b,a))
        # FP64 CPU dot of selected rows isolates matrix arithmetic from RMS.
        ids = torch.linspace(0,m-1,min(m,16),device='cuda').long()
        exact_dot = flat[ids].cpu().double() @ fn.cpu().double().T
        raw_error = (raw[ids].cpu().double()-exact_dot).abs().max().item()
        permutation = torch.arange(m-1,-1,-1,device='cuda')
        shuffled = flat[permutation].contiguous(); other = torch.empty_like(raw)
        op(shuffled,fn,other)
        permutation_exact = torch.equal(other[permutation].view(torch.int32),raw.view(torch.int32))
        del shuffled,other
        graphs={}
        for arm, call in [('A',baseline),('B',candidate)]:
            for _ in range(3):call()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):call()
            graphs[arm]=g
        graphs['B'].replay(); original=b.clone()
        for _ in range(100):graphs['B'].replay()
        replay_exact=torch.equal(original.view(torch.int32),b.view(torch.int32))
        samples=[]
        for cycle in range(3):
            for arm in ('A','B','B','A'):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):graphs[arm].replay()
                end.record();end.synchronize()
                samples.append(dict(cycle=cycle,arm=arm,ms=start.elapsed_time(end)/5))
        med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
        result=dict(name=name,checks=checks,raw_fp64_max_abs=raw_error,
            permutation_exact=permutation_exact,replay100_exact=replay_exact,median_ms=med,samples=samples,
            scratch_bytes=raw.numel()*4+(scratch.numel()*4 if split else 0))
        case['candidates'].append(result);save()
        print(m,name,med,'error',checks[-1],'perm',permutation_exact,'replay',replay_exact,flush=True)
        assert all(c['finite'] for c in checks) and permutation_exact and replay_exact
        del g,graphs,original
        if split: del scratch
    del x,residual,post,comb,hidden,flat,partials,a,b,raw
    torch.cuda.empty_cache()
report['status']='complete';save()
