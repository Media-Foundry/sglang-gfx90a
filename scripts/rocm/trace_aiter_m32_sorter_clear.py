#!/usr/bin/env python3
"""Trace/sentinel probe for AIter M32 A4 sorter output-buffer clearing."""
import torch
from torch.profiler import ProfilerActivity, profile
from aiter.fused_moe import moe_sorting

torch.cuda.set_device(0)
ids=torch.randint(0,256,(32,6),device='cuda',dtype=torch.int32)
w=torch.rand((32,6),device='cuda',dtype=torch.float32)
buf=torch.full((32,4096),7.0,device='cuda',dtype=torch.bfloat16)
for _ in range(3): out=moe_sorting(ids,w,256,4096,torch.bfloat16,block_size=4,moe_buf=buf)
torch.cuda.synchronize()
print('SENTINEL before=7 after_nonzero=',int(torch.count_nonzero(buf).item()),'numel=',buf.numel())
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20): out=moe_sorting(ids,w,256,4096,torch.bfloat16,block_size=4,moe_buf=buf)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20,max_name_column_width=200))
