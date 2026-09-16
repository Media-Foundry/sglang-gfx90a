"""First-layer-only FP32 mix table derived from original token embeddings.

No service integration. Applicable only before the first layer for plain token
embedding inputs: it is NOT valid for input_embeds, adapters or later residuals.
Production would build from TP-local vocabulary shards, not replicate weights.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from safetensors import safe_open
from sglang.kernels.ops.layernorm.gfx90a_mhc_pre_mix import _jit_gfx90a_mhc_pre_mix_module

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'oracle.json';assert not target.exists()
torch.set_num_threads(1);torch.set_grad_enabled(False)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
model=Path('/home/pc/models/modelscope')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def load(name):
    with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).cuda()
embed=load('embed.weight');fn=load('layers.0.hc_attn_fn').float().reshape(24,16384).contiguous()
assert embed.shape==(129280,4096) and embed.dtype==torch.bfloat16
digest=lambda x:hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
trace=repo/'.agents/experiments/dsv4_prefill_tail_contract_20260917/trace-v3-serial/trace'
meta=torch.load(trace/'layer-0-rank-0-metadata.pt',weights_only=False)
ids=meta['input_ids'][:64].cuda()
def suffix(stage):
    record=json.loads((trace/f'layer-0-rank-0-{stage}.json').read_text())
    p=trace/record['values_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==record['file_sha256']
    raw=torch.load(p,weights_only=False)['samples']
    rows=[meta['metadata']['sampled_rows'].index(i) for i in range(64)]
    return raw[rows].contiguous().view(getattr(torch,record['dtype'].removeprefix('torch.'))).cuda()
residual=embed[ids].unsqueeze(1).repeat(1,4,1)
assert torch.equal(residual,suffix('attn_residual'))
assert digest(fn)==json.loads((trace/'layer-0-rank-0-hc_attn_fn.json').read_text())['sha256']
module=_jit_gfx90a_mhc_pre_mix_module()
table=torch.empty((len(embed),1,24),device='cuda',dtype=torch.float32)
report=dict(status='running',scope=__doc__,vocab_size=len(embed),table_bytes=table.numel()*table.element_size(),
    build_activation_bound_bytes=1024*4*4096*2,embed_sha256=digest(embed),fn_sha256=digest(fn),
    live_residual_exact=True,checks=[],timings=[],
    sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),repo/'python/sglang/kernels/jit/csrc/gemm/gfx90a_mhc_pre_mix.cuh')})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def lookup(token_ids):return table[token_ids]
def direct(token_ids):
    x=embed[token_ids].unsqueeze(1).repeat(1,4,1)
    out=torch.empty((len(token_ids),1,24),device='cuda',dtype=torch.float32)
    (module.run_m64 if len(token_ids)==64 else module.run)(x,fn,out,1e-6)
    return out
def graph(op):
    for _ in range(3):result=op()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):result=op()
    return g,result
def timing(g):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):g.replay()
    b.record();b.synchronize();return a.elapsed_time(b)/20
save()
try:
    start=time.perf_counter()
    for offset in range(0,len(embed),1024):
        end=min(offset+1024,len(embed))
        x=embed[offset:end].unsqueeze(1).repeat(1,4,1)
        module.run(x,fn,table[offset:end],1e-6)
        # Whole vocabulary: validate against the other native geometry as well.
        other=torch.empty_like(table[offset:end]);module.run_m64(x,fn,other,1e-6)
        assert torch.equal(table[offset:end],other),(offset,'native variants')
    torch.cuda.synchronize();report['build_and_validate_wall_s']=time.perf_counter()-start
    report['whole_vocab_native_variants_exact']=True
    assert torch.equal(lookup(ids),suffix('unfused_mhc_mixes'))
    report['service_mixes_exact']=True
    manifest=json.loads((repo/'.agents/experiments/dsv4_ck_route_producer_service_20260917/B/inputs.json').read_text())
    real_ids=torch.tensor([t for r in manifest['requests'] for t in r['input_ids']],device='cuda')
    for m in (1,64,65,1024,8192,32576,32768):
        token_ids=real_ids[:m].clone()
        result=lookup(token_ids);ref=direct(token_ids);assert torch.equal(result,ref)
        perm=torch.arange(m-1,-1,-1,device='cuda')
        assert torch.equal(lookup(token_ids[perm])[perm],ref)
        g,y=graph(lambda:lookup(token_ids))
        for mutation in range(100):
            token_ids.copy_(real_ids[(torch.arange(m,device='cuda')+mutation*997)%len(real_ids)])
            g.replay();torch.cuda.synchronize()
            assert torch.equal(y,direct(token_ids)),(m,mutation)
        report['checks'].append(dict(m=m,initial_exact=True,permutation_exact=True,graph_mutations_exact=100))
        # Direct includes embedding gather/repeat; also isolate premix on ready residual.
        x=embed[token_ids].unsqueeze(1).repeat(1,4,1)
        raw=torch.empty((m,1,24),device='cuda',dtype=torch.float32)
        control,_=graph(lambda:(module.run(x,fn,raw,1e-6),raw)[1])
        samples={'control_ready_residual':[],'table_lookup':[]}
        for _ in range(4):
            for name,which in (('control_ready_residual',control),('table_lookup',g),('table_lookup',g),('control_ready_residual',control)):
                samples[name].append(timing(which))
        report['timings'].append(dict(m=m,samples_ms=samples,median_ms={k:statistics.median(v) for k,v in samples.items()}))
        save();print('CASE',m,report['timings'][-1]['median_ms'],flush=True)
        del g,control,y,x,raw
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
