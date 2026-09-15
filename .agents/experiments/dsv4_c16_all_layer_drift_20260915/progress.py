"""Read completed JSON only; never infer liveness from these files."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
for arm in ('A','A2','B2'):
    data=root/arm/'data'
    if not data.exists():continue
    complete=[];divergent=[]
    for layer in range(43):
        for stage in ('attn_norm','ffn_input','ffn_out'):
            paths=[data/f'layer-{layer}-rank-{rank}-{stage}.json' for rank in range(8)]
            if not all(p.exists() for p in paths):continue
            try:records=[json.loads(p.read_text()) for p in paths]
            except json.JSONDecodeError:continue  # writer may still be active
            if stage=='ffn_out':complete.append(layer)
            if len({r.get('sha256') for r in records})!=1:
                divergent.append(dict(layer=layer,stage=stage,
                    hashes=[r.get('sha256') for r in records]))
    print(arm,'all-eight-rank completed layers',complete,
          'replicated-boundary mismatches',divergent[:3],flush=True)
for left,right in (('A','A2'),('A2','B2')):
    differences=[]
    for layer in range(43):
        for rank in range(8):
            for stage in ('attn_residual','attn_norm','attn_core','attn_out','ffn_input',
                          'ffn_topk_ids','ffn_topk_weights','ffn_routed','ffn_out'):
                paths=[root/arm/'data'/f'layer-{layer}-rank-{rank}-{stage}.json' for arm in (left,right)]
                if not all(p.exists() for p in paths):continue
                try:a,b=[json.loads(p.read_text()) for p in paths]
                except json.JSONDecodeError:continue
                if a.get('sha256')!=b.get('sha256'):differences.append((layer,rank,stage))
    print(left,'vs',right,'provisional differences (metadata not yet validated):',differences[:5],flush=True)
    print('  first FFN-output differences:',[r for r in differences if r[2]=='ffn_out'][:8],flush=True)
