"""Exact-M query16 versus runtime-M query16; no production source changes.

Reuses the integrated score/Top-K mutation fixture. Compiler assembly hashes
are grouped by all non-M metadata to verify reuse across row counts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch

os.environ.setdefault('SGLANG_OPT_USE_TRITON_INDEXER_FULL','1')
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER','3')
from sglang.kernels.ops.attention.dsv4 import gfx90a_indexer_query_reuse as prod
from sglang.srt.layers.attention.dsv4 import indexer
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from candidate import reuse_runtime_m

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mutations',type=int,default=100)
p.add_argument('--replays',type=int,default=100)
p.add_argument('--sizes',type=int,nargs='+',default=[8192,32768,65536])
p.add_argument('--irregular',action='store_true',help='Also check odd M at the same W/page strides as large prefill')
p.add_argument('--integrated',action='store_true',help='Use the production wrapper runtime_m keyword and imported kernel')
a=p.parse_args();assert not a.output.exists()
fixture_output=a.output.with_suffix('.fixture.json');assert not fixture_output.exists()
base=prod.prefill_query_reuse4
if a.integrated:
    from sglang.kernels.ops.attention.dsv4 import gfx90a_indexer_runtime_m as integrated
    runtime_kernel=integrated.reuse_runtime_m
else:
    runtime_kernel=reuse_runtime_m
compiled_records={}

class RuntimeKernel:
    def __getitem__(self,grid):
        def launch(*args,**kw):
            assert args[10]==16
            compiled=runtime_kernel[grid](*args,**kw)
            if compiled is not None:
                key=tuple(args[6:13])
                if key not in compiled_records:
                    compiled_records[key]=dict(m=args[6],width=args[7],pages=args[8],
                        page_stride=args[9],group=args[10],block_s=args[11],shuffle=args[12],
                        amdgcn_sha256=hashlib.sha256(compiled.asm['amdgcn'].encode()).hexdigest(),
                        hsaco_sha256=(hashlib.sha256(compiled.asm['hsaco']).hexdigest()
                                      if isinstance(compiled.asm.get('hsaco'),bytes) else None),
                        compiled_object_id=id(compiled),cache_hash=getattr(compiled,'hash',None),
                        regs=compiled.n_regs,spills=compiled.n_spills,lds=compiled.metadata.shared)
            return compiled
        return launch

def candidate(*args,**kwargs):
    if a.integrated:
        with patch.object(integrated,'reuse_runtime_m',RuntimeKernel()):
            return base(*args,query_group_size=16,runtime_m=True,**kwargs)
    with patch.object(prod,'reuse',RuntimeKernel()):
        return base(*args,query_group_size=16,**kwargs)

def control(q,cache,w,lens,pages,metadata,width,clean_logits,**kwargs):
    assert kwargs==dict(skip_trivial_topk=512,skip_empty_tiles=True)
    out=base(q,cache,w,lens,pages,width,query_group_size=16,
        block_s=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S.get(),
        preshuffle_tile=(indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE
                        if indexer.aiter_can_use_preshuffle_paged_mqa() else 0),
        dot_fp16=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get(),
        fp8_fnuz=indexer.is_fp8_fnuz())
    assert out is not None
    return out

root=Path(__file__).resolve().parent
fixture_path=root.parent/'dsv4_c16_indexer_qreuse_20260915/screen.py'
spec=importlib.util.spec_from_file_location('query_reuse_fixture',fixture_path)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
indexer.fp8_paged_mqa_logits_torch=control
prod.prefill_query_reuse4=candidate
sys.argv=[str(fixture_path),'--runtime','--bq','4','--mutations',str(a.mutations),
          '--replays',str(a.replays),'--sizes',*map(str,a.sizes),'--output',str(fixture_output)]
fixture.main()
data=json.loads(fixture_output.read_text());assert data['status']=='complete'
if a.irregular:
    maxm=65536
    q=torch.randn(maxm,1,64,128,device='cuda').to(indexer.FP8_DTYPE)
    w=torch.randn(maxm,64,device='cuda')
    lens=((torch.arange(maxm,device='cuda')%8192+1)//4).int()
    pages=((torch.arange(maxm,device='cuda')//8192)[:,None]*32
           +torch.arange(32,device='cuda')[None,:]).int()
    cache=torch.empty(256,64,1,132,device='cuda',dtype=torch.uint8)
    values=torch.randn(256*64,128,device='cuda')
    loc=torch.arange(256*64,device='cuda',dtype=torch.int32)
    triton_fused_store_indexer(values,cache.view(256,8448),loc,64)
    options=dict(block_s=16,preshuffle_tile=(indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE
                 if indexer.aiter_can_use_preshuffle_paged_mqa() else 0),
                 dot_fp16=False,fp8_fnuz=indexer.is_fp8_fnuz())
    extra=[]
    for m in (32766,32767,65535):
        outputs=[];graphs=[]
        def stage(runtime):
            x=(q[:m],cache,w[:m],lens[:m],pages[:m],2048)
            scores=candidate(*x,**options) if runtime else base(*x,query_group_size=16,**options)
            logical=torch.empty(m,512,device='cuda',dtype=torch.int32)
            physical=torch.empty_like(logical)
            topk_transform_512(scores,lens[:m],pages[:m],physical,64,logical)
            return scores,logical,physical
        for runtime in (False,True):
            stage(runtime)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs.append(stage(runtime))
            graphs.append(graph)
        for mutation in range(a.mutations):
            q[:m].copy_(torch.randn(q[:m].shape,device='cuda').to(indexer.FP8_DTYPE))
            w[:m].normal_();values.normal_()
            triton_fused_store_indexer(values,cache.view(256,8448),loc,64)
            for graph in graphs:graph.replay()
            assert torch.equal(outputs[0][0].view(torch.int32),outputs[1][0].view(torch.int32)),('odd scores',m,mutation)
            assert torch.equal(outputs[0][1],outputs[1][1]),('odd logical',m,mutation)
            assert torch.equal(outputs[0][2],outputs[1][2]),('odd physical',m,mutation)
        extra.append(dict(m=m,mutations=a.mutations,all_exact=True))
    data['irregular_same_metadata']=extra
records=list(compiled_records.values())
large=[r for r in records if r['width']==2048]
data.update(scope=__doc__,integrated_entry=a.integrated,control_bq=16,candidate_bq=16,runtime_m=True,
            compiler_records=records,
            same_large_shape_binary=len({r['amdgcn_sha256'] for r in large})==1,
            same_large_shape_cached_object=len({r['compiled_object_id'] for r in large})==1,
            same_large_shape_hsaco=(all(r['hsaco_sha256'] is not None for r in large)
                                   and len({r['hsaco_sha256'] for r in large})==1),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            candidate_sha256=hashlib.sha256((root/'candidate.py').read_bytes()).hexdigest())
if a.integrated:
    data['integrated_kernel_sha256']=hashlib.sha256(Path(integrated.__file__).read_bytes()).hexdigest()
a.output.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(binary_reuse=data['same_large_shape_binary'],compiler_records=records,
                     summary=[(r['batch'],r['medians_us'],r['stage_speedup']) for r in data['cases']]),indent=2))
