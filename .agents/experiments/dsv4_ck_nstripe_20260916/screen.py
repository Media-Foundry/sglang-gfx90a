"""Full unique CK stage2 + all stripe reducers; no service or stage1 timing."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module

root=Path(__file__).resolve().parent
target=root/'screen.json';assert not target.exists()
def module(manifest):
    record=json.loads(Path(manifest).read_text());assert record['status']=='complete'
    assert hashlib.sha256(Path(record['module']).read_bytes()).hexdigest()==record['module_sha256']
    for path,digest in record['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    spec=importlib.util.spec_from_file_location(record['name'],record['module'])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,record
base,base_manifest=module(root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json')
stripe,stripe_manifest=module(json.loads((root/'build.json').read_text())['manifest'])
reducer=load_jit('dsv4_ck_nstripe_reducer',cuda_files=[str(root/'reducer.cuh')],
    cuda_wrappers=[('bf16','sglang::StripeReducer::bf16'),('fp32','sglang::StripeReducer::fp32')],
    extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],
    extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
refmod=fixed_slot_module()
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
torch.set_grad_enabled(False)
report=dict(status='running',pci=pci.value.decode(),scope=__doc__,
    baseline_module_sha256=base_manifest['module_sha256'],stripe_module_sha256=stripe_manifest['module_sha256'],
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'reducer.cuh']},cases=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
for location in ('dsv4_cached_ck_drift_20260915/capture','dsv4_c16_real_ck_replay_20260915/capture-v2'):
    fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
    data={}
    for name in ('weight2','stage2_input','sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights'):
        path=fixture/(name+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
        data[name]=torch.load(path,map_location='cuda',weights_only=True).contiguous()
    inter=data['stage2_input'];m,t,k=inter.shape;assert t==6 and k==256
    ids=torch.empty_like(data['sorted_token_ids'])
    refmod.remap(data['sorted_token_ids'],data['num_valid_ids'],ids,m)
    ref_partial=torch.empty((m*6,4096),device='cuda',dtype=torch.float32)
    ref_out=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
    candidate_out=torch.empty_like(ref_out)
    args=(inter.view(m*6,1,k),data['weight2'],ids,data['sorted_expert_ids'],data['num_valid_ids'],data['sorted_weights'])
    def baseline():
        base.stage2(*args,ref_partial)
        refmod.reduce_vec4(ref_partial.view(m,6,4096),ref_out)
    baseline();assert bool(torch.isfinite(ref_partial).all())
    for width in (128,256,512,1024,2048,4096):
        partial=torch.empty((m*6,width),device='cuda',dtype=torch.float32)
        # Independent producer contract: every stripe's FP32 partial equals the
        # corresponding full-N partial, not merely its final rounded BF16 sum.
        for mutation in range(3):
            if mutation:
                inter.mul_(.999);data['sorted_weights'].mul_(.997)
                baseline()
            candidate_out.fill_(float('nan'))
            for n0 in range(0,4096,width):
                partial.fill_(float('nan'))
                stripe.stage2(*args,partial,n0)
                exact=torch.equal(partial.view(torch.int32),ref_partial[:,n0:n0+width].view(torch.int32))
                if not exact:
                    report.update(status='failed-correctness',failed=dict(m=m,width=width,n0=n0,mutation=mutation,
                        max_abs=float((partial-ref_partial[:,n0:n0+width]).abs().max())))
                    save();raise AssertionError(report['failed'])
                reducer.bf16(partial.view(m,6,width),candidate_out,n0)
            assert torch.equal(candidate_out.view(torch.int16),ref_out.view(torch.int16))
        def candidate():
            for n0 in range(0,4096,width):
                stripe.stage2(*args,partial,n0)
                reducer.bf16(partial.view(m,6,width),candidate_out,n0)
        for _ in range(3):baseline();candidate()
        graphs={}
        for name,call in [('A',baseline),('B',candidate)]:
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):call()
            graphs[name]=g
        for _ in range(100):graphs['B'].replay()
        assert torch.equal(candidate_out.view(torch.int16),ref_out.view(torch.int16))
        timings={}
        for mode in ('eager','graph'):
            records=[]
            for cycle in range(3):
                for name in ('A','B','B','A'):
                    torch.cuda.synchronize()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    call=(baseline if name=='A' else candidate) if mode=='eager' else graphs[name].replay
                    a.record()
                    for _ in range(3):call()
                    b.record();b.synchronize();records.append(dict(cycle=cycle,arm=name,ms=a.elapsed_time(b)/3))
            med={arm:statistics.median(r['ms'] for r in records if r['arm']==arm) for arm in ('A','B')}
            timings[mode]=dict(median_ms=med,speedup=med['A']/med['B'],samples=records)
        report['cases'].append(dict(m=m,width=width,scratch_bytes=partial.numel()*4,fixture=str(fixture),
            mutations3_partial_exact=True,final_bf16_exact=True,replay100_exact=True,timings=timings))
        save();print('STRIPE',m,width,{mode:v['median_ms'] for mode,v in timings.items()},flush=True)
        del graphs,g,partial
    del inter,data,ref_partial,ref_out,candidate_out,args,ids
    torch.cuda.empty_cache()
report['status']='complete';save()
