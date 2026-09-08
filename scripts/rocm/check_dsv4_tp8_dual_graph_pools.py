#!/usr/bin/env python3
"""Eight-rank AIter IPC registration and isolated M32 graph-pool oracle.

No model/HTTP integration. Run torchrun --standalone --nproc-per-node=8.
"""
import datetime
import json
import os

import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=120))
    assert dist.get_world_size()==8
    assert torch.cuda.get_device_properties(rank).gcnArchName.startswith('gfx90a')
    comm=CustomAllreduce(dist.group.WORLD,torch.device('cuda',rank))
    assert not comm.disabled
    x1=torch.empty((1,4096),device='cuda',dtype=torch.bfloat16)
    x32=torch.empty((32,4096),device='cuda',dtype=torch.bfloat16)
    pattern=torch.arange(4096,device='cuda',dtype=torch.float32).remainder_(31)
    for x in (x1,x32):x.copy_(pattern*.125+rank)
    base_pool=torch.cuda.graph_pool_handle()
    other_pool=torch.cuda.graph_pool_handle()
    assert base_pool!=other_pool
    graphs={};outputs={}
    before=torch.cuda.memory_allocated()
    reserved_before=torch.cuda.memory_reserved()
    free_before=torch.cuda.mem_get_info()[0]
    # Mirrors the outer GroupCoordinator capture context: register all graph
    # addresses only after all shapes/pools have completed capture.
    with comm.capture():
        for name,x,pool in (('a32',x32,base_pool),('c1',x1,base_pool),('b32',x32,other_pool)):
            def forward():
                produced=x*1.0  # input to AR is a graph-pool allocation
                out=comm.custom_all_reduce(produced,use_new=(name=='c1'))
                assert out is not None
                return out
            for _ in range(2):forward()
            torch.cuda.synchronize();dist.barrier()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,pool=pool):out=forward()
            graphs[name]=g;outputs[name]=out
    torch.cuda.synchronize();dist.barrier()
    after=torch.cuda.memory_allocated()
    reserved_after=torch.cuda.memory_reserved()
    free_after=torch.cuda.mem_get_info()[0]
    assert after-before < 64*1024**2, ('unexpected small-oracle workspace',after-before)
    pointers={name:out.data_ptr() for name,out in outputs.items()}
    assert len(set(pointers.values()))==3
    identities={name:id(g) for name,g in graphs.items()}
    expected={}
    for name,g in graphs.items():
        g.replay();torch.cuda.synchronize()
        expected[name]=(pattern+28).expand_as(outputs[name]).to(torch.bfloat16)
        assert torch.equal(outputs[name].view(torch.int16),expected[name].view(torch.int16))
    order=('a32','b32','b32','a32','c1','b32','c1','a32')
    pools={'a32':base_pool,'c1':base_pool,'b32':other_pool}
    snapshots={key:out.clone() for key,out in outputs.items()}
    shared_pool_overwrites=0
    for iteration in range(1000):
        name=order[iteration%len(order)]
        x=x1 if name=='c1' else x32
        shift=iteration%4
        x.copy_((pattern+shift)*.125+rank)
        graphs[name].replay();torch.cuda.synchronize()
        expected[name]=(pattern+shift+28).expand_as(outputs[name]).to(torch.bfloat16)
        # Different shapes in the original shared pool may overwrite old
        # outputs once their consumer has finished. Only the added pool must
        # remain isolated. Always validate the graph actually replayed.
        for key,out in outputs.items():
            assert out.data_ptr()==pointers[key] and id(graphs[key])==identities[key]
            if key==name:
                assert torch.equal(out.view(torch.int16),expected[key].view(torch.int16)), (rank,iteration,key)
            elif pools[key]!=pools[name]:
                assert torch.equal(out.view(torch.int16),snapshots[key].view(torch.int16)), (rank,iteration,key,'cross-pool overwrite')
            else:
                shared_pool_overwrites+=int(not torch.equal(out.view(torch.int16),snapshots[key].view(torch.int16)))
        for key,out in outputs.items():snapshots[key].copy_(out)
    dist.barrier()
    stats=[None]*8
    dist.all_gather_object(stats,dict(rank=rank,graph_allocated_delta=after-before,
        graph_reserved_delta=reserved_after-reserved_before,
        device_free_drop=free_before-free_after,
        output_pointers=pointers,graph_identity_stable=True,
        shared_pool_overwrites=shared_pool_overwrites))
    if rank==0:print(json.dumps(dict(status='passed',mutated_replays=1000,
        graph_count=3,pool_count=2,cross_pool_outputs_preserved=True,ranks=stats)),flush=True)
    comm.close()
    dist.destroy_process_group()


if __name__=='__main__':main()
