"""Compare complete query-producer inputs and KV in logical request order."""
import argparse
import json
from pathlib import Path
import torch
import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='layer1-prepare')
    parser.add_argument('--layer',type=int,default=1)
    args=parser.parse_args()
    assert '/' not in args.run_name and args.run_name not in ('.','..')
    root=Path(__file__).resolve().parent/args.run_name
    assert (root/'complete.json').exists()
    cases=json.loads((root/'inputs.json').read_text())['requests']
    lookup={analyze.digest(v['input_ids']):i for i,v in enumerate(cases)}
    slices={};metadata={}
    for arm in ('A1','B1','A2'):
        directory=root/('trace-'+arm)
        ids=torch.load(directory/f'layer_{args.layer}_rank_0_input_ids.pt',weights_only=True).tolist()
        positions=torch.load(directory/f'layer_{args.layer}_rank_0_positions.pt',weights_only=True).tolist()
        starts=[i for i,p in enumerate(positions) if p==0];ends=starts[1:]+[len(ids)]
        slices[arm]={}
        for first,last in zip(starts,ends,strict=True):
            case=lookup[analyze.digest(ids[first:last])]
            assert positions[first:last]==list(range(last-first))
            slices[arm][case]=(first,last)
        metadata[arm]=dict(m=len(ids),cases=list(slices[arm]))
    result=dict(layer=args.layer,metadata=metadata,comparisons={})
    for a,b in [('A1','A2'),('A1','B1')]:
        records=[]
        common=sorted(set(slices[a])&set(slices[b]))
        for rank in range(8):
            stages=('prepare_full_input','prepare_full_qkv_a','prepare_full_kv') if rank==0 else ('prepare_full_kv',)
            for stage in stages:
                x=torch.load(root/('trace-'+a)/f'layer_{args.layer}_rank_{rank}_{stage}.pt',weights_only=True)
                y=torch.load(root/('trace-'+b)/f'layer_{args.layer}_rank_{rank}_{stage}.pt',weights_only=True)
                for case in common:
                    lo,hi=slices[a][case];jl,jh=slices[b][case]
                    xx=x[lo:hi];yy=y[jl:jh]
                    assert xx.shape==yy.shape
                    delta=(xx.float()-yy.float()).abs()
                    row_changed=torch.any(xx!=yy,dim=1)
                    changed_rows=row_changed.nonzero().flatten().tolist()
                    tail=delta[-128:]
                    item=dict(rank=rank,stage=stage,case=case,shape=list(xx.shape),
                        changed_elements=int(torch.count_nonzero(delta)),
                        changed_rows=changed_rows,max_abs=float(delta.max()),
                        last128_changed=int(torch.count_nonzero(tail)),last128_max_abs=float(tail.max()))
                    records.append(item)
                    print(a,b,rank,stage,case,'rows',len(changed_rows),'elements',item['changed_elements'],
                          'max',item['max_abs'],'last128',item['last128_changed'],flush=True)
        result['comparisons'][a+'-'+b]=records
    (root/'prepare-summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
