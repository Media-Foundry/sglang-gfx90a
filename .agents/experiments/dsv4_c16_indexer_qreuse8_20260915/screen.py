"""Compare the accepted query4 chain against query8/16, without service edits.

Reuse the integrated fixture, including strided mixed-page M17 and Top-K checks.
The Python wrapper keeps its existing name; only the test-local launch proxy
changes BQ and grid in arm B. Arm A calls the saved production query4 wrapper.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault('SGLANG_OPT_USE_TRITON_INDEXER_FULL','1')
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER','3')
import triton
from sglang.kernels.ops.attention.dsv4 import gfx90a_indexer_query_reuse as prod
from sglang.srt.layers.attention.dsv4 import indexer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bq',type=int,choices=(8,16),required=True)
p.add_argument('--direct',action='store_true',help='Exercise the integrated group-size argument, not launch proxy')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mutations',type=int,default=100)
p.add_argument('--replays',type=int,default=100)
p.add_argument('--sizes',type=int,nargs='+',default=[8192,32768])
a=p.parse_args();assert not a.output.exists()
original_kernel=prod.reuse
original_wrapper=prod.prefill_query_reuse4
resources=[]

class Candidate:
    def __getitem__(self,grid):
        def launch(*args,**kw):
            args=list(args)
            assert args[10]==4 and grid[0]==triton.cdiv(args[6],4)
            args[10]=a.bq
            compiled=original_kernel[(triton.cdiv(args[6],a.bq),grid[1])](*args,**kw)
            if compiled is not None and not resources:
                resources.append(dict(bq=args[10],m=args[6],regs=compiled.n_regs,
                                      spills=compiled.n_spills,lds=compiled.metadata.shared))
            return compiled
        return launch

def candidate(*args,**kwargs):
    if a.direct:
        return original_wrapper(*args,query_group_size=a.bq,**kwargs)
    with patch.object(prod,'reuse',Candidate()):
        return original_wrapper(*args,**kwargs)

def control(q,cache,w,lens,pages,metadata,width,clean_logits,**kwargs):
    assert kwargs==dict(skip_trivial_topk=512,skip_empty_tiles=True)
    result=original_wrapper(q,cache,w,lens,pages,width,
        block_s=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S.get(),
        preshuffle_tile=(indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE
                        if indexer.aiter_can_use_preshuffle_paged_mqa() else 0),
        dot_fp16=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get(),
        fp8_fnuz=indexer.is_fp8_fnuz())
    assert result is not None
    return result

root=Path(__file__).resolve().parent
fixture_path=root.parent/'dsv4_c16_indexer_qreuse_20260915/screen.py'
spec=importlib.util.spec_from_file_location('query_reuse_fixture',fixture_path)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
indexer.fp8_paged_mqa_logits_torch=control
prod.prefill_query_reuse4=candidate
sys.argv=[str(fixture_path),'--runtime','--bq','4','--mutations',str(a.mutations),
          '--replays',str(a.replays),'--sizes',*map(str,a.sizes),'--output',str(a.output)]
fixture.main()
data=json.loads(a.output.read_text())
assert data['status']=='complete' and data['ragged_mixed_page_mutations_exact']==a.mutations
data.update(scope=__doc__,direct=a.direct,bq=a.bq,control_bq=4,candidate_bq=a.bq,resources=resources,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            production_kernel_sha256=hashlib.sha256(Path(prod.__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(control_bq=4,candidate_bq=a.bq,resources=resources,
                     summary=[(r['batch'],r['medians_us'],r['stage_speedup']) for r in data['cases']]),indent=2))
