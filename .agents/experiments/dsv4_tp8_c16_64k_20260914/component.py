"""TP8-shape CK: one 64K stage vs two 32K stages, without model-service claims."""
import json
import os
from pathlib import Path
import statistics
import subprocess
from unittest.mock import patch

import torch

root=Path(__file__).resolve().parent
target=root/'component.json'
assert not target.exists() and os.environ.get('HIP_VISIBLE_DEVICES')=='4'
gpu=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in gpu for p in g.get('process_list',[]))
os.environ.update(SGLANG_DSV4_DEBUG_TP8_CK_64K='1',
    SGLANG_DSV4_GFX90A_BF16_CK_STAGE2_FP32='1',SGLANG_DSV4_GFX90A_BF16_CK_BLOCK64_V1='1',
    AITER_DSV4_DEBUG_SHUFFLE_BF16_WEIGHTS='1',SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT='0')
from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import gfx90a_bf16_ck_moe, bf16_ck_prefill_max_rows
with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_TP8_CK_64K':'0'}):
    assert bf16_ck_prefill_max_rows(256)==bf16_ck_prefill_max_rows(512)==36864
assert bf16_ck_prefill_max_rows(256)==65536 and bf16_ck_prefill_max_rows(512)==36864
torch.manual_seed(20260914)
m,h,e,t,i=65536,4096,256,6,256
x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)
w13=torch.randint(0,256,(e,2*i,h//2),device='cuda',dtype=torch.uint8)
w2=torch.randint(0,256,(e,h,i//2),device='cuda',dtype=torch.uint8)
s13=torch.randint(118,121,(e,2*i,h//32),device='cuda',dtype=torch.uint8)
s2=torch.randint(118,121,(e,h,i//32),device='cuda',dtype=torch.uint8)
weights=torch.softmax(torch.randn((m,t),device='cuda'),dim=-1)
records=[]
for kind in ('balanced','skewed'):
    ids=torch.arange(m*t,device='cuda',dtype=torch.int32).reshape(m,t)%e
    if kind=='skewed':
        ids=torch.cat((torch.tensor([0,1,2],device='cuda',dtype=torch.int32).expand(m,3),
                       3+torch.arange(m*3,device='cuda',dtype=torch.int32).reshape(m,3)%253),dim=1)
    for size in (65536,65533):
        def stage(lo,hi):return gfx90a_bf16_ck_moe(x[lo:hi],ids[lo:hi],weights[lo:hi],w13,s13,w2,s2)
        def separate():return stage(0,32768),stage(32768,size)
        def joint():return stage(0,size)
        a,b=separate();ref=torch.cat((a,b));del a,b
        y=joint()
        assert torch.isfinite(y).all() and torch.isfinite(ref).all()
        delta=y.float()-ref.float()
        relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float()))
        record=dict(kind=kind,m=size,max_abs=float(delta.abs().max()),relative_l2=relative,
                    exact_fraction=float((y==ref).float().mean()))
        assert relative<0.005,record
        del y,ref,delta
        samples={'A':[],'B':[]}
        # Full stage, including sorter, expansion, CK stage1/stage2 and final cast.
        for arm in ['A','B','B','A']*3:
            fn=separate if arm=='A' else joint
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();out=fn();end.record();end.synchronize()
            samples[arm].append(begin.elapsed_time(end));del out
        record['samples_ms']=samples
        record['median_ms']={arm:statistics.median(v) for arm,v in samples.items()}
        record['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        records.append(record);print(record,flush=True)
target.write_text(json.dumps(dict(gpu_before=gpu,records=records),indent=2)+'\n')
