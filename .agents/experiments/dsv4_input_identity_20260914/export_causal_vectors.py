"""Publish only small causal activation vectors, never model weights."""
import gzip
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'causal-vectors.json.gz'
    assert not out.exists()
    runs={}
    for run in ('all-ranks','combined-ffn','both-ar-l0'):
        data={}
        for arm,row in [('A1',24576),('B1',24575)]:
            directory=ROOT/run/('trace-'+arm)
            ranks=[]
            for rank in range(8):
                prefix=f'layer_0_rank_{rank}_'
                rows=torch.load(directory/(prefix+'sample_rows.pt'),weights_only=True)
                indices=(rows==row).nonzero().flatten();assert indices.numel()==1
                record={}
                for stage in ('wo_b_partial','wo_b','ffn_partial','ffn_out'):
                    p=directory/(prefix+stage+'.pt')
                    if not p.exists():continue
                    t=torch.load(p,weights_only=True)[int(indices.item())].contiguous()
                    assert t.dtype==torch.bfloat16 and t.shape==(4096,)
                    record[stage]=dict(shape=list(t.shape),bf16_bits=t.view(torch.uint16).tolist())
                ranks.append(record)
            data[arm]=dict(flat_row=row,layer=0,case=15,position=0,ranks=ranks)
        runs[run]=data
    payload=dict(format='unsigned integer BF16 bit patterns; activation vectors only',runs=runs)
    with gzip.open(out,'wt') as f:json.dump(payload,f,separators=(',',':'))
    with gzip.open(out,'rt') as f:loaded=json.load(f)
    checks=[]
    for run in ('combined-ffn','both-ar-l0'):
        for arm in ('A1','B1'):
            r=loaded['runs'][run][arm]['ranks']
            def tensor(rank,stage):return torch.tensor(r[rank][stage]['bf16_bits'],dtype=torch.uint16).view(torch.bfloat16)
            ref=sum(tensor(rank,'ffn_partial').double() for rank in range(8)).bfloat16()
            actual=tensor(0,'ffn_out')
            changed=int(torch.count_nonzero(ref!=actual))
            if run=='both-ar-l0':assert changed==0
            checks.append(dict(run=run,arm=arm,changed_vs_fp64=changed))
    (ROOT/'causal-vectors-manifest.json').write_text(json.dumps(dict(
        bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        validation=checks),indent=2)+'\n')
    print('Exported activation-only vectors:',out.stat().st_size,'bytes;',checks)


if __name__=='__main__':main()
