"""Default-off native prefill compressor input/output witnesses; no math changes."""
from sglang.kernels.ops.debug.dsv4_prepare_dump import make_prepare_dump
import torch

def _dump(compressor,batch):
    from sglang.srt.runtime_context import get_parallel
    return make_prepare_dump(compressor.layer_id,get_parallel().attn_tp_rank,
                             batch,batch.positions)

def projection(compressor,x,batch,value):
    dump=_dump(compressor,batch)
    if dump is not None:
        name='index' if compressor.is_in_indexer else 'core'
        for suffix,tensor in (('input',x),('weight',compressor.wkv_gate.weight),('projection',value)):
            dump(f'prepare_compressor_{name}_{suffix}',tensor,full=True,rank0_only=True)
    return value

def storage(compressor,batch,value,cache,locations,plan,*,bf16_store):
    from sglang.srt.runtime_context import get_parallel
    if compressor.is_in_indexer or get_parallel().attn_tp_rank!=0:
        return
    dump=_dump(compressor,batch)
    if dump is not None:
        raw=plan[1].view(torch.int32)
        assert not plan.is_decode and bf16_store
        assert cache.ndim==2 and cache.dtype==torch.uint8 and cache.shape[1]==1024
        # CompressPlan gives ragged_id its full uint32 word (large prefill).
        ragged=raw[:,1].long() & 0xFFFFFFFF
        selected=locations[ragged.long()].long()
        assert selected.numel()==value.shape[0]
        assert int(selected.min())>=0 and int(selected.max())<cache.shape[0]
        stored=cache.index_select(0,selected).view(torch.bfloat16)
        for suffix,tensor in (('pooled',value),('result',stored),('locations',selected),('plan',raw)):
            dump(f'prepare_compressor_core_{suffix}',tensor,full=True,rank0_only=True)
