"""Direct CK-layout decode versus production LDS preshuffle; byte-exact gate."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import _jit_dequant,_logical_a16w4_scales

root=Path(__file__).resolve().parent
bit_path='--bits' in sys.argv
row_path='--rows' in sys.argv
assert not(bit_path and row_path)
suffix='_rows' if row_path else '_bits' if bit_path else ''
target=root/('rows-screen.json' if row_path else 'bits-screen.json' if bit_path else 'screen.json');assert not target.exists()
mod=load_jit('dsv4_dequant_direct_screen',cuda_files=[str(root/'direct.cuh')],
    cuda_wrappers=[(n+s,'sglang::DirectDequant::'+n+s) for s in ('','_bits','_rows') for n in ('gate','down','tiny')],
    extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],extra_cuda_cflags=['-O3'])
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',pci=pci.value.decode(),cases=[],exhaustive=[],
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'direct.cuh']})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
tinyref=_jit_dequant(2,16,32,416)
bits=torch.arange(16,device='cuda',dtype=torch.int32)
w=(bits|((15-bits)<<4)).to(torch.uint8).repeat(2,16,1).contiguous()
a=torch.empty((2,16,32),device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a)
for first in range(0,256,32):
    s=torch.arange(first,first+32,device='cuda',dtype=torch.int32).to(torch.uint8).view(2,16,1)
    tinyref.run_shuffled(w,s,a);getattr(mod,'tiny'+suffix)(w,s,b,416)
    exact=torch.equal(a.view(torch.int16),b.view(torch.int16))
    report['exhaustive'].append(dict(scale_start=first,all16_nibbles=True,byte_exact=exact));save()
    assert exact,('exhaustive scales',first)
del w,s,a,b
fixture=root.parent/'dsv4_c16_real_ck_replay_20260915/capture-v2/fixture'
meta=json.loads((fixture/'manifest.json').read_text())
for name,wn,sn,refname,n,k,gate in [('gate','raw_w13','raw_s13','weight13',512,4096,True),
                                  ('down','raw_w2','raw_s2','weight2',4096,256,False)]:
    data={}
    for key in (wn,sn,refname):
        path=fixture/(key+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[key]['file_sha256']
        data[key]=torch.load(path,map_location='cuda',weights_only=True).contiguous()
    w=data[wn].view(torch.uint8)
    s=_logical_a16w4_scales(data[sn],256,n,k//32,gate_up=gate) if meta['scales_shuffled'] else data[sn].view(torch.uint8).view(256,n,k//32)
    a=torch.empty((256,n,k),device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a)
    ref=_jit_dequant(256,n,k,1664);fn=getattr(mod,name+suffix)
    ref.run_shuffled(w,s,a);fn(w,s,b,1664)
    assert torch.equal(a.view(torch.int16),data[refname].view(torch.int16)),(name,'capture layout')
    assert torch.equal(a.view(torch.int16),b.view(torch.int16))
    for mutation in range(10):
        w.bitwise_xor_(1);s.add_(1)
        b.fill_(float('nan'));ref.run_shuffled(w,s,a);fn(w,s,b,1664)
        assert torch.equal(a.view(torch.int16),b.view(torch.int16)),(name,mutation)
    for blocks in (416,832,1664):
        def baseline():ref.run_shuffled(w,s,a)
        def candidate():fn(w,s,b,blocks)
        for _ in range(3):baseline();candidate()
        graphs={}
        for arm,call in [('A',baseline),('B',candidate)]:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            graphs[arm]=graph
        for _ in range(100):graphs['B'].replay()
        assert torch.equal(a.view(torch.int16),b.view(torch.int16))
        samples=[]
        for cycle in range(3):
            for arm in ('A','B','B','A'):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(5):graphs[arm].replay()
                end.record();end.synchronize();samples.append(dict(cycle=cycle,arm=arm,ms=begin.elapsed_time(end)/5))
        med={arm:statistics.median(v['ms'] for v in samples if v['arm']==arm) for arm in ('A','B')}
        report['cases'].append(dict(name=name,blocks=blocks,shape=[256,n,k],median_ms=med,
            mutation10_byte_exact=True,replay100_byte_exact=True,samples=samples))
        save();print('DEQUANT',name,blocks,med,flush=True)
        del graphs,graph
    del w,s,a,b,data
    torch.cuda.empty_cache()
report['status']='complete';save()
