#!/usr/bin/env python3
"""Isolated TP8 AIter old/new AR screen; direct HIP input, fixed graph outputs.

Run with torchrun --standalone --nproc-per-node=8, on idle service GPUs.
No library rebuilds, global environment changes, or production modifications.
"""
import argparse
import datetime
import json
import os
import statistics

import torch
import torch.distributed as dist
import aiter
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=32, choices=(1,32))
    p.add_argument('--mutations', type=int, default=100)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--rounds', type=int, default=5)
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=180))
    assert dist.get_world_size() == 8
    ar = CustomAllreduce(dist.group.WORLD, torch.device('cuda',rank), max_size=1024*1024)
    assert not ar.disabled
    storage = aiter.allocate_meta_buffer(args.rows*4096*2)
    x = storage.view(torch.bfloat16).view(args.rows,4096)
    ar.register_buffer(x)
    outputs = [torch.empty_like(x),torch.empty_like(x)]
    graphs = []
    x.fill_(rank+1)
    for use_new,y in zip((True,False),outputs):
        ar.all_reduce(x,out=y,registered=True,use_new=use_new)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ar.all_reduce(x,out=y,registered=True,use_new=use_new)
        ar.register_graph_buffers()
        graphs.append(g)
    exact = 0
    max_abs = 0.
    stable = True
    torch.manual_seed(20908+rank)
    for i in range(args.mutations):
        x.normal_()
        if i%4 == 0:
            x.copy_(torch.randint(-8,9,x.shape,device=x.device).to(x.dtype))
        torch.cuda.synchronize()
        dist.barrier()
        for g in graphs:g.replay()
        torch.cuda.synchronize()
        exact += int(torch.equal(*outputs))
        max_abs=max(max_abs,float((outputs[0]-outputs[1]).abs().max()))
        reference=outputs[1].clone()
        for _ in range(10):graphs[1].replay()
        torch.cuda.synchronize()
        stable = stable and torch.equal(reference,outputs[1])
    reports=[None]*8
    dist.all_gather_object(reports,dict(rank=rank,exact=exact,max_abs=max_abs,stable=stable))
    samples=[[],[]]
    for _ in range(args.rounds):
        for arm in (0,1,1,0):
            for _ in range(20):graphs[arm].replay()
            torch.cuda.synchronize()
            dist.barrier()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iters):graphs[arm].replay()
            end.record();end.synchronize()
            local=begin.elapsed_time(end)*1000/args.iters
            values=[None]*8
            dist.all_gather_object(values,local)
            samples[arm].append(max(values))
    if rank==0:
        print(json.dumps(dict(rows=args.rows,bytes=x.numel()*2,correctness=reports,
                              rankmax_samples_us=samples,
                              medians_us=[statistics.median(v) for v in samples],
                              trimmed_us=[statistics.mean(sorted(v)[1:-1]) for v in samples])),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
