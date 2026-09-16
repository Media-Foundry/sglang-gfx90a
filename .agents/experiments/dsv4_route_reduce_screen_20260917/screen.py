"""Three route-reducer variants on historical real MoE inputs; never service data.

Only run after all service regressions stop. Both isolated reducer and complete
metadata/stage1/stage2/reduce chains are timed; sort/dequant/allocation excluded.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from sglang.kernels.jit.utils import load_jit

root=Path(__file__).resolve().parent
repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'screen-v1.json')
args=parser.parse_args()
assert not args.output.exists()
torch.set_grad_enabled(False);torch.set_num_threads(1)
assert torch.cuda.device_count()==1
assert torch.cuda.mem_get_info()[0]>50*1024**3,'Requires idle GCD; no overlapping service'
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',scope=__doc__,pci=pci.value.decode(),cases=[],modules={},
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__),root/'reduce.cuh',
        repo/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_ck_route_producer.cuh',
        repo/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_ck_fixed_slot.cuh']})
def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
def exact(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
def load_module(path,label):
    record=json.loads(path.read_text());assert record['status']=='complete'
    assert hashlib.sha256(Path(record['module']).read_bytes()).hexdigest()==record['module_sha256']
    for name,digest in record['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    spec=importlib.util.spec_from_file_location(record['name'],record['module'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report['modules'][label]=dict(manifest=str(path),sha256=record['module_sha256'])
    return module
def graph(call):
    for _ in range(3):call()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):call()
    return g
def abba(a,b):
    samples=[]
    for cycle in range(3):
        for arm in ('A','B','B','A'):
            torch.cuda.synchronize()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(3):(a if arm=='A' else b).replay()
            end.record();end.synchronize()
            samples.append(dict(cycle=cycle,arm=arm,ms=begin.elapsed_time(end)/3))
    med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
    return dict(median_ms=med,speedup=med['A']/med['B'],samples=samples)

save()
try:
    producer=load_module(root.parent/'dsv4_ck_route_producer_20260916/build-v1/route/manifest.json','stage1')
    ck=load_module(root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json','stage2')
    names=('v4wave','v8','v8wave')
    mod=load_jit('dsv4_route_reduce_screen',cuda_files=[str(root/'reduce.cuh')],
        cuda_wrappers=[('metadata','sglang::RouteMetadata::run'),
            ('reference','sglang::RouteMajor::bf16'),('reference_f32','sglang::RouteMajor::fp32')]
            +[(name,'sglang::RouteReduceScreen::'+name) for name in names]
            +[(name+'_f32','sglang::RouteReduceScreen::'+name+'_f32') for name in names],
        extra_include_paths=[str(repo/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
    for location in ('dsv4_cached_ck_drift_20260915/capture','dsv4_c16_real_ck_replay_20260915/capture-v2'):
        fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
        cpu={}
        for name in ('hidden','weight13','weight2','sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights'):
            path=fixture/(name+'.pt')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
            cpu[name]=torch.load(path,map_location='cpu',weights_only=True).contiguous()
        m=cpu['hidden'].size(0);capacity=cpu['sorted_token_ids'].numel();valid=int(cpu['num_valid_ids'][0])
        encoded=cpu['sorted_token_ids'][:valid].long();token=encoded&0xffffff;slot=encoded>>24
        live=(token<m)&(slot>=0)&(slot<6)
        assert torch.equal(torch.sort(token[live]*6+slot[live]).values,torch.arange(m*6))
        data={k:v.cuda() for k,v in cpu.items()}
        hidden=data['hidden'];w13=data['weight13'];w2=data['weight2']
        ids=data['sorted_token_ids'];se=data['sorted_expert_ids'];nv=data['num_valid_ids'];weights=data['sorted_weights']
        inter=torch.empty((capacity,256),dtype=torch.bfloat16,device='cuda')
        identity=torch.empty_like(ids);inverse=torch.empty((m,6),dtype=torch.int32,device='cuda')
        partial=torch.empty((capacity,4096),dtype=torch.float32,device='cuda')
        ref=torch.empty((m,4096),dtype=torch.bfloat16,device='cuda');out=torch.empty_like(ref)
        ref32=torch.empty_like(ref,dtype=torch.float32);out32=torch.empty_like(ref32)
        def produce():
            mod.metadata(ids,nv,identity,inverse)
            producer.stage1(hidden,w13,ids,se,nv,inter)
            ck.stage2(inter.view(capacity,1,256),w2,identity,se,nv,weights,partial)
        def full(name,dst):
            produce();getattr(mod,name)(partial,inverse,dst)
        checks=[]
        for mutation in range(4):
            if mutation:hidden.mul_(.999);weights.mul_(.997)
            if mutation==3:
                ids[:valid].copy_(ids[:valid].view(-1,64).flip(0).reshape(-1))
                weights[:valid].copy_(weights[:valid].view(-1,64).flip(0).reshape(-1))
                se[:valid//64].copy_(se[:valid//64].flip(0))
            partial.fill_(float('nan'));inverse.fill_(-1);produce()
            assert bool(((inverse>=0)&(inverse<capacity)).all())
            mod.reference(partial,inverse,ref);mod.reference_f32(partial,inverse,ref32)
            assert bool(torch.isfinite(ref32).all())
            for name in names:
                out.fill_(float('nan'));out32.fill_(float('nan'))
                getattr(mod,name)(partial,inverse,out);getattr(mod,name+'_f32')(partial,inverse,out32)
                assert exact(out,ref) and exact(out32,ref32),(m,name,mutation)
                checks.append(dict(mutation=mutation,variant=name,bf16_exact=True,fp32_exact=True))
        # Stable-address input mutation and repeated graph replay, both outputs.
        for name in names:
            gb=graph(lambda:getattr(mod,name)(partial,inverse,out))
            gf=graph(lambda:getattr(mod,name+'_f32')(partial,inverse,out32))
            for rep in range(100):
                if rep%10==0:partial.add_(.0001)
                mod.reference(partial,inverse,ref);mod.reference_f32(partial,inverse,ref32)
                out.fill_(float('nan'));out32.fill_(float('nan'));gb.replay();gf.replay()
                assert exact(out,ref) and exact(out32,ref32),(m,name,rep)
            del gb,gf
        for name in ('hidden','sorted_token_ids','sorted_expert_ids','sorted_weights'):data[name].copy_(cpu[name])
        produce()
        timings={}
        for mode in ('reducer_only','producer_chain'):
            a=graph(lambda:mod.reference(partial,inverse,ref)) if mode=='reducer_only' else graph(lambda:full('reference',ref))
            results={}
            for name in names:
                b=graph(lambda:getattr(mod,name)(partial,inverse,out)) if mode=='reducer_only' else graph(lambda:full(name,out))
                a.replay();b.replay();assert exact(out,ref),(m,name,mode)
                results[name]=abba(a,b)
                del b
            timings[mode]=results;del a
        report['cases'].append(dict(m=m,capacity=capacity,checks=checks,
            graph_replays_per_variant_per_dtype=100,timings=timings))
        save();print('CASE',m,{k:{n:v['median_ms'] for n,v in values.items()} for k,values in timings.items()},flush=True)
        del cpu,data,hidden,w13,w2,ids,se,nv,weights,inter,identity,inverse,partial,ref,out,ref32,out32
        torch.cuda.empty_cache()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
