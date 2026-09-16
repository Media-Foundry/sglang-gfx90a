"""Exact route-major storage oracle; complete pack+CK+fixed-reduction ABBA.

Uses the accepted unique-Set CK binary unchanged. No production selector edits.
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
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'screen.json')
args=parser.parse_args()
target=args.output;assert not target.exists()
torch.set_grad_enabled(False)
assert torch.cuda.mem_get_info()[0]>50*1024**3, 'Requires an idle GCD, no concurrent service'
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',scope=__doc__,pci=pci.value.decode(),cases=[],
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'route_major.cuh']})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    record=json.loads((root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json').read_text())
    assert record['status']=='complete'
    assert hashlib.sha256(Path(record['module']).read_bytes()).hexdigest()==record['module_sha256']
    for p,h in record['sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    spec=importlib.util.spec_from_file_location(record['name'],record['module'])
    ck=importlib.util.module_from_spec(spec);spec.loader.exec_module(ck)
    report['baseline_module_sha256']=record['module_sha256']
    mod=load_jit('dsv4_ck_route_major',cuda_files=[str(root/'route_major.cuh')],
        cuda_wrappers=[('pack','sglang::RouteMajor::pack'),('bf16','sglang::RouteMajor::bf16'),('fp32','sglang::RouteMajor::fp32')],
        extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
    refmod=fixed_slot_module()
    for location in ('dsv4_cached_ck_drift_20260915/capture','dsv4_c16_real_ck_replay_20260915/capture-v2'):
        fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
        cpu={}
        for name in ('weight2','stage2_input','sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights'):
            p=fixture/(name+'.pt');assert hashlib.sha256(p.read_bytes()).hexdigest()==meta[name]['file_sha256']
            cpu[name]=torch.load(p,map_location='cpu',weights_only=True).contiguous()
        m,t,k=cpu['stage2_input'].shape;assert t==6 and k==256
        a=meta['stage2_args'];assert a['block_m']==64 and a['quant_type']==0 and not a['kernel']
        valid=int(cpu['num_valid_ids'][0]);encoded=cpu['sorted_token_ids'][:valid].long()
        tokens=encoded&0xffffff;slots=encoded>>24;live=(tokens<m)&(slots<6)
        virtual=tokens[live]*6+slots[live]
        assert torch.equal(torch.sort(virtual).values,torch.arange(m*6))
        data={name:value.cuda() for name,value in cpu.items()}
        inter=data['stage2_input'];ids=data['sorted_token_ids'];nv=data['num_valid_ids']
        se=data['sorted_expert_ids'];weights=data['sorted_weights'];w2=data['weight2']
        capacity=ids.numel();assert valid%64==0 and valid<=capacity
        remapped=torch.empty_like(ids)
        packed=torch.empty((capacity,256),dtype=torch.bfloat16,device='cuda')
        identity=torch.empty_like(ids)
        inverse=torch.empty((m,6),dtype=torch.int32,device='cuda')
        ref_partial=torch.empty((m*6,4096),dtype=torch.float32,device='cuda')
        partial=torch.empty((capacity,4096),dtype=torch.float32,device='cuda')
        ref_out=torch.empty((m,4096),dtype=torch.bfloat16,device='cuda')
        out=torch.empty_like(ref_out)
        def baseline():
            refmod.remap(ids,nv,remapped,m)
            ck.stage2(inter.view(m*6,1,256),w2,remapped,se,nv,weights,ref_partial)
            refmod.reduce_vec4(ref_partial.view(m,6,4096),ref_out)
        def candidate():
            mod.pack(inter,ids,nv,packed,identity,inverse)
            ck.stage2(packed.view(capacity,1,256),w2,identity,se,nv,weights,partial)
            mod.bf16(partial,inverse,out)
        for mutation in range(5):
            if mutation:
                inter.mul_(.999);weights.mul_(.997)
            if mutation==3:
                # Preserve expert-block membership/weights but reverse block order.
                ids[:valid].copy_(ids[:valid].view(-1,64).flip(0).reshape(-1))
                weights[:valid].copy_(weights[:valid].view(-1,64).flip(0).reshape(-1))
                se[:valid//64].copy_(se[:valid//64].flip(0))
            inverse.fill_(-1);partial.fill_(float('nan'));ref_partial.fill_(float('nan'))
            baseline();candidate()
            assert bool(((inverse>=0)&(inverse<capacity)).all())
            idx=inverse.flatten().long()
            assert torch.equal(packed.index_select(0,idx).view(torch.int16),inter.view(m*6,256).view(torch.int16))
            gathered=partial.index_select(0,idx)
            assert bool(torch.isfinite(gathered).all()) and bool(torch.isfinite(ref_partial).all())
            assert torch.equal(gathered.view(torch.int32),ref_partial.view(torch.int32)),('partial',m,mutation)
            assert torch.equal(out.view(torch.int16),ref_out.view(torch.int16)),('BF16',m,mutation)
            del idx,gathered
        # Restore the measured real workload after mutation/permutation checks.
        for name in ('stage2_input','sorted_token_ids','sorted_expert_ids','sorted_weights'):
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
            records=[]
            for cycle in range(3):
                for name in ('A','B','B','A'):
                    torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    call=(baseline if name=='A' else candidate) if mode=='eager' else graphs[name].replay
                    start.record()
                    for _ in range(3):call()
                    end.record();end.synchronize()
                    records.append(dict(cycle=cycle,arm=name,ms=start.elapsed_time(end)/3))
            med={arm:statistics.median(r['ms'] for r in records if r['arm']==arm) for arm in ('A','B')}
            timings[mode]=dict(median_ms=med,speedup=med['A']/med['B'],samples=records)
        report['cases'].append(dict(m=m,fixture=str(fixture),capacity=capacity,valid=valid,
            unique_assignments=m*6,mutations5_partial_bf16_exact=True,expert_block_permutation_exact=True,
            replay100_exact=True,baseline_partial_bytes=ref_partial.numel()*4,
            candidate_partial_bytes=partial.numel()*4,packing_bytes=packed.numel()*2,timings=timings))
        save();print('ROUTE MAJOR',m,{mode:v['median_ms'] for mode,v in timings.items()},flush=True)
        del graphs,graph,cpu,data,inter,ids,nv,se,weights,w2,remapped,packed,identity,inverse,ref_partial,partial,ref_out,out
        torch.cuda.empty_cache()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
