#!/usr/bin/env python3
"""TP8 standalone greedy-tail oracle; no model or production selectors."""
import json
import os
import statistics

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 8
    torch.manual_seed(20260908 + rank)
    for m in (1, 32):
        x = torch.randn((m, 16160), dtype=torch.bfloat16, device='cuda')
        gathered = torch.empty((8*m, 16160), dtype=x.dtype, device='cuda')
        pairs = torch.empty((m, 2), device='cuda')
        all_pairs = torch.empty((8*m, 2), device='cuda')

        def baseline():
            dist.all_gather_into_tensor(gathered, x)
            logits = gathered.view(8,m,16160).permute(1,0,2).reshape(m,129280).float()
            return logits.argmax(-1)

        def candidate():
            score, index = x.max(-1)
            pairs[:,0].copy_(score)
            pairs[:,1].copy_(index + rank*16160)
            dist.all_gather_into_tensor(all_pairs, pairs)
            p = all_pairs.view(8,m,2)
            best = p[:,:,0].max(0).values
            ids = torch.where(p[:,:,0] == best, p[:,:,1], 16777216.)
            return ids.min(0).values.to(torch.int64)

        graphs, outputs = [], []
        for fn in (baseline, candidate):
            for _ in range(3): fn()
            torch.cuda.synchronize(); dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = fn()
            graphs.append(g); outputs.append(out)
        for n in range(100):
            x.normal_()
            if n % 10 == 0: x[:,7] = 20 # cross-rank ties
            if n % 10 == 1: x.fill_(float('-inf'))
            if n % 10 == 2: x[:,11] = float('inf')
            for g in graphs: g.replay()
            torch.cuda.synchronize()
            assert torch.equal(*outputs), (rank,m,n)
        samples = [[],[]]
        for _ in range(5):
            for arm in (0,1,1,0):
                dist.barrier()
                for _ in range(20): graphs[arm].replay()
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100): graphs[arm].replay()
                end.record();end.synchronize()
                duration=torch.tensor([start.elapsed_time(end)*10],device='cuda')
                dist.all_reduce(duration,op=dist.ReduceOp.MAX)
                samples[arm].append(duration.item())
        if rank==0:
            print(json.dumps(dict(m=m,world=8,exact_mutations_per_rank=100,
                samples_rankmax_us=samples,
                trimmed_us=[statistics.mean(sorted(s)[1:-1]) for s in samples])),flush=True)
        # Release captured RCCL references before destroying the communicator.
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        del graphs, outputs
    dist.destroy_process_group()


if __name__=='__main__': main()
