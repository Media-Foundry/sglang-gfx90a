#!/usr/bin/env python3
"""Compare valid identical-prefix rows within an existing DSV4 stage dump."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump-dir',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--rank',type=int,default=0)
    a=p.parse_args()
    torch.set_num_threads(4)
    prefix=json.loads(a.reference.read_text())['teacher_forced'][0]['input_ids']
    stem=a.dump_dir/f'layer_{a.layer}_rank_{a.rank}_'
    def load(name):return torch.load(str(stem)+name+'.pt',map_location='cpu',weights_only=True)
    pos,ids=load('positions').reshape(-1),load('input_ids').reshape(-1)
    starts=[]
    for i in torch.where((pos==0)&(ids==prefix[0]))[0].tolist():
        if ids[i:i+len(prefix)].tolist()==prefix and pos[i:i+len(prefix)].tolist()==list(range(len(prefix))):starts.append(i)
    assert len(starts)>=2,('no repeated complete prefixes',starts,ids.shape)
    indices=torch.tensor([[i+j for j in range(len(prefix))] for i in starts])
    print(json.dumps({'starts':starts,'prefix_length':len(prefix),'rows':len(ids)}),flush=True)
    for name in ('attn_residual','attn_pre_norm','attn_norm','q','attn_core','attn_inverse_rope','wo_a','wo_b_partial','wo_b','attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','attn_hc_post','ffn_norm','ffn_out','ffn_hc_post'):
        path=Path(str(stem)+name+'.pt')
        if not path.exists():continue
        x=load(name)
        if x.shape[0]!=len(ids):continue
        selected=x[indices].float()
        diff=selected-selected[:1]
        print(json.dumps({'stage':name,'shape':list(x.shape),
            'max_abs':diff.abs().max().item(),
            'exact_rows':(diff.reshape(len(starts),len(prefix),-1)==0).all(-1).sum().item(),
            'total_rows':len(starts)*len(prefix),
            'finite':bool(torch.isfinite(selected).all())}),flush=True)


if __name__=='__main__':main()
