"""Default-off runner-owned H16 peer attention experiment; ordinary prefill only."""
import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist


def eligible(stage1, rows, ratio, capacity):
    return stage1 and 8192 <= rows <= capacity and (ratio == 4 or (ratio == 128 and rows >= 16384))


class PeerPrefill:
    def __init__(self, capacity, manifest_path, *, tp=None):
        assert 8192 <= capacity <= 32768
        assert dist.get_world_size() == 8
        assert os.environ.get('HIP_VISIBLE_DEVICES') == '0,1,2,3,4,5,6,7'
        self.rank = dist.get_rank();self.parity = self.rank % 2
        self.device = torch.cuda.current_device();assert self.device == self.rank
        from sglang.srt.distributed import get_tp_group
        self.tp = get_tp_group() if tp is None else tp
        assert self.tp.rank_in_group == self.rank
        members = [None] * 8
        dist.all_gather_object(members, (socket.gethostname(), self.device), group=self.tp.cpu_group)
        assert len({x[0] for x in members}) == 1
        assert [x[1] for x in members] == list(range(8))
        for base in range(0,8,2):
            cpu = dist.new_group([base,base+1],backend='gloo',timeout=datetime.timedelta(minutes=30))
            gpu = dist.new_group([base,base+1],backend='nccl',timeout=datetime.timedelta(minutes=30))
            if self.rank in (base,base+1):self.cpu,self.comm = cpu,gpu
        assert torch.cuda.can_device_access_peer(self.device, self.device ^ 1)
        manifest = json.loads(Path(manifest_path).read_text())
        for key,digest in (('source','source_sha256'),('module','module_sha256')):
            assert hashlib.sha256(Path(manifest[key]).read_bytes()).hexdigest() == manifest[digest]
        spec = importlib.util.spec_from_file_location(manifest['name'],manifest['module'])
        self.ipc = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.ipc)
        self.capacity = capacity;self.bytes = capacity * 8 * 512 * 2 * 2
        self.storage = self.ipc.allocate(self.bytes)
        handles = [None] * 2
        dist.all_gather_object(handles,self.ipc.handle(self.storage),group=self.cpu)
        self.peer_storage = self.ipc.open(handles[1-self.parity],self.bytes)
        self.q,self.out = self.storage.view(torch.bfloat16).view(2,capacity,8,512).unbind(0)
        self.peer_q,self.peer_out = self.peer_storage.view(torch.bfloat16).view(2,capacity,8,512).unbind(0)
        self.sinks = torch.empty(16,device='cuda',dtype=torch.float32)
        self.fence = torch.zeros(1,device='cuda')
        self.audited = set();self.logged = False
        print(f'[TP{self.rank}] H16 peer workspace ready: capacity={capacity} bytes={self.bytes} pair={self.rank//2}',flush=True)

    def audit(self, layer, args, batch):
        from sglang.kernels.ops.debug.dsv4_attention_peer_capture import canonical_csr
        m = len(args['q']);digest = hashlib.sha256()
        for bank,ids in (('unified_kv','kv_indices_prefix'),('kv_extend','kv_indices_extend')):
            ptr = ids.replace('indices','indptr')
            used,canonical,pointers = canonical_csr(args[ids].cpu().numpy(),args[ptr].cpu().numpy(),m)
            digest.update(canonical.tobytes());digest.update(pointers.tobytes())
            if len(used):
                assert int(used.max()) < len(args[bank])
                values = args[bank].index_select(0,torch.from_numpy(used).to(args['q'].device))
                digest.update(values.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        for t in (batch.input_ids,batch.positions):
            digest.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        pair = [None]*2
        dist.all_gather_object(pair,digest.hexdigest(),group=self.cpu)
        assert pair[0] == pair[1], ('peer KV/selection/input mismatch',layer,pair)
        print(f'[TP{self.rank}] H16 peer logical audit: layer={layer} rows={m} hash={pair[0]}',flush=True)

    def forward(self, layer, args, batch, checking=False):
        from sglang.kernels.ops.debug.dsv4_h16_peer_kernel import peer_prefill
        m = len(args['q']);assert 8192 <= m <= self.capacity
        assert args['q'].shape == (m,8,512) and args['q'].dtype == torch.bfloat16
        assert batch.forward_mode.name == 'EXTEND' and not torch.cuda.is_current_stream_capturing()
        if checking and layer not in self.audited:
            self.audit(layer,args,batch);self.audited.add(layer)
        self.q[:m].copy_(args['q'])
        dist.all_gather_into_tensor(self.sinks,args['attn_sink'][:8].contiguous(),group=self.comm)
        dist.all_reduce(self.fence,group=self.comm)
        q0,q1 = (self.q,self.peer_q) if self.parity == 0 else (self.peer_q,self.q)
        o0,o1 = (self.out,self.peer_out) if self.parity == 0 else (self.peer_out,self.out)
        half = (m+1)//2;rows = half if self.parity == 0 else m-half
        peer_prefill[(rows,1)](q0,args['unified_kv'],args['kv_indices_prefix'],args['kv_indptr_prefix'],
            args['kv_extend'],args['kv_indices_extend'],args['kv_indptr_extend'],self.sinks,o0,q1,o1,self.parity*half,
            *self.q.stride(),*args['unified_kv'].stride(),*args['kv_extend'].stride(),*self.out.stride(),
            16,512,args['softmax_scale'],BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=1)
        dist.all_reduce(self.fence,group=self.comm)
        out = self.out[:m]
        if checking:
            from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
            reference = runtime.prefill(**args,num_stages=1)
            assert torch.equal(out.view(torch.uint8),reference.view(torch.uint8)),('H16 peer output mismatch',layer,m)
            print(f'[TP{self.rank}] H16 peer output exact: layer={layer} rows={m}',flush=True)
        if not self.logged:
            print(f'[TP{self.rank}] H16 peer selected: rows={m} check={int(checking)} bytes={self.bytes}',flush=True)
            self.logged = True
        return out

    def close(self):
        # Explicit collective close for controlled callers; service lifetime is
        # runner/process lifetime. Never call a collective from __del__.
        torch.cuda.synchronize();dist.barrier(group=self.cpu)
        self.peer_q=self.peer_out=self.peer_storage=None;gc.collect()
        dist.barrier(group=self.cpu)
        self.q=self.out=self.storage=None;gc.collect()
