"""Real-fixture stage1 producer correctness, then full stage1+stage2 ABBA.

Includes metadata and fixed reduction; excludes weight expansion and sorter.
Requires one idle GCD. No production selector or installed header changes.
"""
import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build',type=Path,default=root/'build-v1')
parser.add_argument('--output',type=Path,default=root/'screen-v1.json')
args=parser.parse_args();target=args.output
assert not target.exists()
torch.set_grad_enabled(False);torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>50*1024**3,'Requires idle GCD without service'
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',scope=__doc__,pci=pci.value.decode(),cases=[],modules={},
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [Path(__file__),root/'metadata.cuh',root.parent/'dsv4_ck_route_major_20260916/route_major.cuh']})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def load_module(manifest,label):
    record=json.loads(manifest.read_text());assert record['status']=='complete'
    assert hashlib.sha256(Path(record['module']).read_bytes()).hexdigest()==record['module_sha256']
    for path,digest in record['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    spec=importlib.util.spec_from_file_location(record['name'],record['module'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report['modules'][label]=dict(manifest=str(manifest),sha256=record['module_sha256'])
    return module
save()
try:
    build=json.loads((args.build/'build.json').read_text());assert build['status']=='complete'
    token=load_module(Path(build['manifests']['token']),'token')
    route=load_module(Path(build['manifests']['route']),'route')
    ck=load_module(root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json','unique_stage2')
    installed=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_b16_dsv4silu_no_mulWeightStage2_')
    report['installed_stage1']=dict(path=installed.__file__,sha256=hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest())
    maps=load_jit('dsv4_route_producer_metadata',cuda_files=[str(root/'metadata.cuh')],
        cuda_wrappers=[('metadata','sglang::RouteMetadata::run'),('reduce','sglang::RouteMajor::bf16')],
        extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
    refmod=fixed_slot_module()
    for location in ('dsv4_cached_ck_drift_20260915/capture','dsv4_c16_real_ck_replay_20260915/capture-v2'):
        fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
        cpu={}
        for name in ('hidden','weight13','weight2','stage1_out','stage2_input','sorted_token_ids',
                     'sorted_expert_ids','num_valid_ids','sorted_weights'):
            path=fixture/(name+'.pt')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
            cpu[name]=torch.load(path,map_location='cpu',weights_only=True).contiguous()
        m,t,i=cpu['stage2_input'].shape;assert (t,i)==(6,256)
        assert torch.equal(cpu['stage1_out'].view(torch.int16),cpu['stage2_input'].view(torch.int16))
        valid=int(cpu['num_valid_ids'][0]);capacity=cpu['sorted_token_ids'].numel()
        assert valid%64==0 and valid<=capacity
        encoded=cpu['sorted_token_ids'][:valid].long();tok=encoded&0xffffff;slot=encoded>>24
        live=(tok<m)&(slot>=0)&(slot<6)
        assert torch.equal(torch.sort(tok[live]*6+slot[live]).values,torch.arange(m*6))
        data={k:v.cuda() for k,v in cpu.items()}
        hidden=data['hidden'];w13=data['weight13'];w2=data['weight2']
        ids=data['sorted_token_ids'];se=data['sorted_expert_ids'];nv=data['num_valid_ids'];weights=data['sorted_weights']
        current=torch.empty((m,6,256),dtype=torch.bfloat16,device='cuda')
        rebuilt=torch.empty_like(current)
        route_inter=torch.empty((capacity,256),dtype=torch.bfloat16,device='cuda')
        identity=torch.empty_like(ids);inverse=torch.empty((m,6),dtype=torch.int32,device='cuda')
        remapped=torch.empty_like(ids)
        ref_partial=torch.empty((m*6,4096),dtype=torch.float32,device='cuda')
        partial=torch.empty((capacity,4096),dtype=torch.float32,device='cuda')
        ref_out=torch.empty((m,4096),dtype=torch.bfloat16,device='cuda');out=torch.empty_like(ref_out)
        def installed_stage1():
            installed.ck_moe_stage1(hidden,w13,w2,ids,se,nv,current,6,meta['stage1_kernel'],
                None,None,64,None,0,3,1,False,None)
        def baseline():
            installed_stage1()
            refmod.remap(ids,nv,remapped,m)
            ck.stage2(current.view(m*6,1,256),w2,remapped,se,nv,weights,ref_partial)
            refmod.reduce_vec4(ref_partial.view(m,6,4096),ref_out)
        def candidate():
            maps.metadata(ids,nv,identity,inverse)
            route.stage1(hidden,w13,ids,se,nv,route_inter)
            ck.stage2(route_inter.view(capacity,1,256),w2,identity,se,nv,weights,partial)
            maps.reduce(partial,inverse,out)
        checks=[]
        for mutation in range(4):
            if mutation:
                hidden.mul_(.999);weights.mul_(.997)
            if mutation==3:
                ids[:valid].copy_(ids[:valid].view(-1,64).flip(0).reshape(-1))
                weights[:valid].copy_(weights[:valid].view(-1,64).flip(0).reshape(-1))
                se[:valid//64].copy_(se[:valid//64].flip(0))
            for tensor in (current,rebuilt,route_inter,ref_partial,partial):tensor.fill_(float('nan'))
            inverse.fill_(-1)
            baseline();candidate()
            token.stage1(hidden,w13,ids,se,nv,rebuilt.view(m*6,256))
            assert torch.equal(rebuilt.view(torch.int16),current.view(torch.int16)),('rebuilt-vs-installed',m,mutation)
            if mutation==0:
                assert torch.equal(current.view(torch.int16),data['stage1_out'].view(torch.int16)),('installed-vs-capture',m)
            assert bool(((inverse>=0)&(inverse<capacity)).all())
            index=inverse.flatten().long()
            gathered=route_inter.index_select(0,index).view(m,6,256)
            assert bool(torch.isfinite(gathered).all())
            assert torch.equal(gathered.view(torch.int16),current.view(torch.int16)),('route-stage1',m,mutation)
            del gathered
            gathered=partial.index_select(0,index)
            assert bool(torch.isfinite(gathered).all())
            assert torch.equal(gathered.view(torch.int32),ref_partial.view(torch.int32)),('stage2-partial',m,mutation)
            assert torch.equal(out.view(torch.int16),ref_out.view(torch.int16)),('final',m,mutation)
            del gathered,index
            checks.append(dict(mutation=mutation,stage1_exact=True,partial_exact=True,final_exact=True))
        for name in ('hidden','sorted_token_ids','sorted_expert_ids','sorted_weights'):
            data[name].copy_(cpu[name])
        for _ in range(3):baseline();candidate()
        graphs={}
        for name,call in [('A',baseline),('B',candidate)]:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            graphs[name]=graph
        for _ in range(100):graphs['B'].replay()
        assert torch.equal(out.view(torch.int16),ref_out.view(torch.int16))
        timings={}
        for mode in ('eager','graph'):
            samples=[]
            for cycle in range(3):
                for arm in ('A','B','B','A'):
                    torch.cuda.synchronize()
                    call=(baseline if arm=='A' else candidate) if mode=='eager' else graphs[arm].replay
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(3):call()
                    end.record();end.synchronize()
                    samples.append(dict(cycle=cycle,arm=arm,ms=begin.elapsed_time(end)/3))
            med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
            timings[mode]=dict(median_ms=med,speedup=med['A']/med['B'],samples=samples)
        report['cases'].append(dict(m=m,capacity=capacity,valid=valid,checks=checks,
            captured_stage1_exact=True,replay100_exact=True,timings=timings,
            extra_intermediate_bytes=(capacity-m*6)*256*2,extra_partial_bytes=(capacity-m*6)*4096*4))
        save();print('FULL PRODUCER CHAIN',m,{k:v['median_ms'] for k,v in timings.items()},flush=True)
        del graphs,graph,cpu,data,hidden,w13,w2,ids,se,nv,weights,current,rebuilt,route_inter,identity,inverse,remapped,ref_partial,partial,ref_out,out
        torch.cuda.empty_cache()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
