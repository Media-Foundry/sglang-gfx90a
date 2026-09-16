"""Numeric deltas on ALL 64 sampled suffix rows of request0, layer0.

Capture order is not causal order: ffn_mhc_post/comb were produced BEFORE
attention, although the existing model dumps them after attn_out.
"""
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
target=root/'trace-values.json';assert not target.exists()
stages=('attn_residual','attn_pre_norm','attn_norm','q','attn_core','wo_a','wo_b',
        'ffn_mhc_post','ffn_mhc_comb','ffn_input','ffn_out')
results=[]
for rank in range(8):
    by_arm=[]
    for arm in ('mixed','serial'):
        d=root/f'trace-{arm}'/'trace';stem=f'layer-0-rank-{rank}'
        meta=torch.load(d/(stem+'-metadata.pt'),weights_only=False)
        assert meta['metadata']['prefix_lens'][0]==32768
        assert meta['metadata']['extend_lens'][0]==64
        assert meta['positions'][:64].tolist()==list(range(32768,32832))
        sample_indices=[meta['metadata']['sampled_rows'].index(i) for i in range(64)]
        tensors={}
        for stage in stages:
            record=json.loads((d/(stem+'-'+stage+'.json')).read_text())
            dtype=getattr(torch,record['dtype'].removeprefix('torch.'))
            payload=torch.load(d/record['values_file'],weights_only=False)
            tensors[stage]=payload['samples'][sample_indices].contiguous().view(dtype).float()
        by_arm.append(tensors)
    for stage in stages:
        x,y=[arm[stage] for arm in by_arm];assert x.shape==y.shape
        diff=(x-y).abs();assert torch.isfinite(diff).all()
        results.append(dict(rank=rank,layer=0,stage=stage,rows=64,elements=x.numel(),
                            different_elements=int((diff!=0).sum()),
                            max_abs=float(diff.max()),mean_abs=float(diff.mean())))
result=dict(status='complete',diagnostic_only=True,results=results,
    caveat='ffn_mhc_post/comb are earlier-produced attention input coefficients, not downstream Q effects')
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([x for x in results if x['rank']==0],indent=2))
