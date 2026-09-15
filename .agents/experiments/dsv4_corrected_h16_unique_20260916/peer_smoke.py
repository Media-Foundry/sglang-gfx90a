"""H16 direct-peer smoke with corrected, genuinely different per-rank sink shards."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from sglang.kernels.ops.debug.dsv4_h16_peer import PeerPrefill
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

root=Path(__file__).resolve().parent;target=root/'peer-smoke.json';assert not target.exists()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('gloo')
fixture=root.parent/'dsv4_local_sink_20260916/check/capture'
meta=json.loads((fixture/f'rank-{rank}.json').read_text());path=fixture/f'rank-{rank}.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['file_sha256']
data=torch.load(path,map_location='cpu',weights_only=True)
args={k:v.cuda() for k,v in data.items() if k not in ('output','input_ids','positions')}
args['softmax_scale']=meta['softmax_scale'];captured=data['output'].cuda()
batch=SimpleNamespace(forward_mode=SimpleNamespace(name='EXTEND'),
    input_ids=data['input_ids'].cuda(),positions=data['positions'].cuda())
worker=PeerPrefill(32768,root.parent/'dsv4_h16_service_20260916/ipc-manifest.json',
    tp=SimpleNamespace(rank_in_group=rank,cpu_group=dist.group.WORLD))
checks=[]
for i in range(10):
    if i:
        args['q'].mul_(.97+rank*.001);args['attn_sink'].add_(.1*(rank+1))
        args['unified_kv'].mul_(.98);args['kv_extend'].mul_(.98)
    worker.out.fill_(float('nan'))
    out=worker.forward(20,args,batch,checking=i==0)
    reference=runtime.prefill(**args,num_stages=1)
    exact=torch.equal(out.view(torch.uint8),reference.view(torch.uint8))
    assert exact and bool(torch.isfinite(out).all())
    if i==0:assert torch.equal(out.view(torch.uint8),captured.view(torch.uint8))
    checks.append(dict(iteration=i,byte_exact=exact))
records=[None]*8;dist.all_gather_object(records,checks)
del out;worker.close();dist.barrier()
if rank==0:
    target.write_text(json.dumps(dict(status='complete',per_rank=records,
        fixture=str(fixture),distinct_sink_mutation=True,explicit_close=True),indent=2)+'\n')
    print('CORRECTED PEER SMOKE COMPLETE',flush=True)
dist.destroy_process_group()
