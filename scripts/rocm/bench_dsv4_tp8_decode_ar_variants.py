#!/usr/bin/env python3
"""Isolated TP8 AIter old/new AR screen; direct HIP input, fixed graph outputs.

Run with torchrun --standalone --nproc-per-node=8, on idle service GPUs.
No library rebuilds, global environment changes, or production modifications.
Rows 64/128 screen DSpark 512-KiB/1-MiB payloads; the independent geometry
shim remains restricted to its validated M32 contract.
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
    p.add_argument('--rows', type=int, default=32, choices=(1,32,64,128))
    p.add_argument('--mutations', type=int, default=100)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--candidate-blocks', type=int, choices=(4,8,12,16,24,32,64,80))
    p.add_argument('--dspark-two-stage', action='store_true',
                   help='independent M64/M128 new AIter two-stage grid oracle')
    p.add_argument('--chain-length', type=int, default=0, choices=(0,32))
    p.add_argument('--chain-replays', type=int, default=1000)
    p.add_argument('--chain-mixed-tiers', action='store_true',
                   help='alternate target M128 tuned grid and installed M64 fallback in the graph chain')
    p.add_argument('--shim-baseline', action='store_true',
                   help='compare both geometries through the validated shim, isolating grid effects')
    args = p.parse_args()
    assert args.mutations > 0 and args.iters > 0 and args.rounds > 0
    assert not args.shim_baseline or args.candidate_blocks is not None
    assert not args.chain_length or args.candidate_blocks is not None
    assert not args.dspark_two_stage or (args.candidate_blocks is not None and args.rows in (64,128))
    assert not args.chain_mixed_tiers or (args.chain_length and args.dspark_two_stage and args.rows == 128)
    assert 1 <= args.chain_replays <= 10000
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=180))
    assert dist.get_world_size() == 8
    ar = CustomAllreduce(dist.group.WORLD, torch.device('cuda',rank), max_size=1024*1024)
    assert not ar.disabled
    geometry = None
    baseline_blocks = min(80, args.rows) if args.dspark_two_stage else 16
    if args.candidate_blocks is not None:
        if args.dspark_two_stage:
            from sglang.kernels.ops.debug.gfx90a_tp8_dspark_ar_oracle import module, run
        else:
            assert args.rows == 32 and args.candidate_blocks <= 32
            from sglang.kernels.ops.debug.gfx90a_tp8_ar_geometry_oracle import module, run
        # Validate the shared Signal layout before using the independent shim.
        assert module().signal_bytes() == aiter.meta_size()
        geometry = run
    storage = aiter.allocate_meta_buffer(args.rows*4096*2)
    x = storage.view(torch.bfloat16).view(args.rows,4096)
    ar.register_buffer(x)
    outputs = [torch.empty_like(x),torch.empty_like(x)]
    graphs = []
    x.fill_(rank+1)
    for arm,(use_new,y) in enumerate(zip((True,False),outputs)):
        def forward():
            if geometry is not None and (arm == 1 or args.shim_baseline):
                geometry(ar,x,y,args.candidate_blocks if arm == 1 else baseline_blocks)
            else:
                ar.all_reduce(x,out=y,registered=True,
                              use_new=use_new if geometry is None else args.dspark_two_stage)
        forward()
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            forward()
        ar.register_graph_buffers()
        graphs.append(g)
    exact = 0
    max_abs = 0.
    stable = True
    integer_failures = 0
    # Bounded integers have an exactly representable sum in BF16 on all
    # eight ranks. Unlike comparing two implementations alone, this catches
    # shared IPC-address errors and stale data in both implementations.
    position = torch.arange(x.numel(), device=x.device).view_as(x)
    torch.manual_seed(20908+rank)
    for i in range(args.mutations):
        x.normal_()
        if i%4 == 0:
            x.copy_(((position + rank * 3 + i) % 17 - 8).to(x.dtype))
            expected = sum((position + peer * 3 + i) % 17 - 8
                           for peer in range(8)).to(x.dtype)
        torch.cuda.synchronize()
        dist.barrier()
        for g in graphs:g.replay()
        torch.cuda.synchronize()
        exact += int(torch.equal(*outputs))
        if i%4 == 0:
            integer_failures += int(any(not torch.equal(y, expected) for y in outputs))
        max_abs=max(max_abs,float((outputs[0]-outputs[1]).abs().max()))
        reference=outputs[1].clone()
        for _ in range(10):graphs[1].replay()
        torch.cuda.synchronize()
        stable = stable and torch.equal(reference,outputs[1])
    reports=[None]*8
    dist.all_gather_object(reports,dict(rank=rank,exact=exact,max_abs=max_abs,
                                      stable=stable,integer_failures=integer_failures))
    assert all(r['exact'] == args.mutations and r['stable'] and r['max_abs'] == 0
               and r['integer_failures'] == 0 for r in reports), reports
    chain_report = None
    if args.chain_length:
        chain_rows = [64 if args.chain_mixed_tiers and step % 2 else args.rows
                      for step in range(args.chain_length)]
        chain_outputs = [torch.empty_like(x[:rows]) for rows in chain_rows]
        chain = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(chain):
            for step, out in enumerate(chain_outputs):
                # Distinct rank-local inputs at every collective: missing exit
                # handshakes can no longer hide behind an unchanged input.
                x.fill_(rank + step)
                if args.chain_mixed_tiers and step % 2:
                    # A draining/changed verify tier returns to installed AR.
                    # This shares the registered base pointer, with a shorter
                    # contiguous view, and exercises cross-grid handshakes.
                    ar.all_reduce(x[:chain_rows[step]], out=out,
                                  registered=True, use_new=True)
                else:
                    geometry(ar, x, out, args.candidate_blocks)
        ar.register_graph_buffers()
        for _ in range(args.chain_replays):
            chain.replay()
        torch.cuda.synchronize()
        chain_exact = all(torch.equal(out, torch.full_like(out, 28 + 8 * step))
                          for step, out in enumerate(chain_outputs))
        witnesses = [None] * 8
        dist.all_gather_object(witnesses, dict(rank=rank, exact=chain_exact))
        assert all(r['exact'] for r in witnesses), witnesses
        chain_report = dict(length=args.chain_length, replays=args.chain_replays,
                            rows=chain_rows, mixed_tiers=args.chain_mixed_tiers,
                            output_bytes=sum(out.numel()*out.element_size() for out in chain_outputs),
                            witnesses=witnesses,
                            limitation='all per-step outputs checked after final replay, not every intermediate replay')
        del chain, chain_outputs
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
                              candidate_blocks=args.candidate_blocks,
                              dspark_two_stage=args.dspark_two_stage,
                              mutating_chain=chain_report,
                              baseline=(f'new two-stage blocks{baseline_blocks}' if args.dspark_two_stage else
                                        'new' if geometry is None else
                                        'shim legacy blocks16' if args.shim_baseline else 'legacy blocks16'),
                              rankmax_samples_us=samples,
                              medians_us=[statistics.median(v) for v in samples],
                              trimmed_us=[statistics.mean(sorted(v)[1:-1] if len(v) > 2 else v)
                                          for v in samples])),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
