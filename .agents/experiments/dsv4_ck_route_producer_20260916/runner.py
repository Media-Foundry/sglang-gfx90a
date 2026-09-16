"""Isolated explicit route-major stages after the SAME AIter sorter.

Not production dispatch. Caller owns already-expanded BF16 layer weights;
per-call temporaries have stream-local PyTorch lifetimes, no global hooks.
"""
import torch
import aiter.fused_moe as fused


class RouteProducer:
    def __init__(self,stage1,stage2,maps):
        self.stage1,self.stage2,self.maps=stage1,stage2,maps

    def __call__(self,hidden,ids,weights,w13,w2,out=None):
        m,h=hidden.shape
        assert 8192<=m<=36864 and h==4096 and ids.shape==(m,6)
        assert hidden.dtype==torch.bfloat16 and w13.shape==(256,512,4096)
        assert w2.shape==(256,4096,256) and w2.dtype==w13.dtype==torch.bfloat16
        si,sw,se,nv,out=fused.moe_sorting(ids,weights,256,4096,torch.bfloat16,64,
                                        None,None,0,out)
        capacity=si.numel()
        inter=torch.empty((capacity,256),device=hidden.device,dtype=torch.bfloat16)
        identity=torch.empty_like(si)
        inverse=torch.empty((m,6),device=hidden.device,dtype=torch.int32)
        partial=torch.empty((capacity,4096),device=hidden.device,dtype=torch.float32)
        self.maps.metadata(si,nv,identity,inverse)
        self.stage1.stage1(hidden,w13,si,se,nv,inter)
        self.stage2.stage2(inter.view(capacity,1,256),w2,identity,se,nv,sw,partial)
        self.maps.reduce(partial,inverse,out)
        return out
